import numpy as np
import multiprocessing
import h5py
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import scipy
import functools
from joblib import Parallel, delayed
from BeadDataFile import *
from tqdm import tqdm


ncores=6

def start_process():
    '''Function ran when starting new multithreading processes'''
    print('Starting', multiprocessing.current_process().name)
    
def average_1darray(arr, numtoaverage):
    numextra = len(arr) % numtoaverage
    return arr[:-numextra].reshape(-1, numtoaverage).mean(axis=1)
    
def lin_detrend(vals):
    '''Performs linear detrending on a 1D array/list of values'''
    slope, intercept, _, _, _ = scipy.stats.linregress(np.array(range(0, len(vals))), vals)

    # Calculate fitted values
    fitted_values = slope * np.array(range(0, len(vals))) + intercept

    # Subtract fitted values from original data
    residuals = vals - fitted_values
    return residuals

def findclosestoset(series, setseries):
    '''Given a series and a list of target values, returns a 1D array of indices corresponding to the eleemnts in the series closest to each target value. Usually used to find target frequencies in FFT results.'''
    return np.array([np.argmin(np.abs(series-setser)) for setser in setseries])

#The following are helper functions to pull specific data from h5 datasets. Note that these do not work if the file is already open.

def getsamplingrate(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        samplingrate = f['auxdata']['samplingrate'][()]
    return samplingrate

def getsubsetmap(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        subsetmap = f['auxdata']['subsetmap'][:]
    return subsetmap

def getnormscale(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        image0 = f['auxdata']['normalizationscale'][()]
    return image0

def getimage0(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        image0 = f['cameradata']['arrays'][0]
    return image0

def getlength(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        length = len(f['cameradata']['arrays'])
    return length

def subsetframe(sourcedata, frame, normalize, normalfactor):
    '''Multithreaded function used in multisubset'''
    if normalize:
        return np.uint8(np.round(normalfactor*sourcedata[frame]/np.sum(sourcedata[frame])))
    else:
        return sourcedata[frame]
    
def newendaxes(dims):
    return tuple([slice(None)]+[np.newaxis]*dims)
    
def multisubset(sourcefilename, targetfilename, frame, datarange = (0,np.inf), normalize=True):
    '''Given a source h5 file, and an array of boolean values corresponding to the image shape,
    creates a new h5 file at a given file path only containing the pixels corresponding to true
    values in the boolean array. Can also be used to normalize each frame by sum.
    
    Note: loads entire file into RAM. Avoid datasets >48000 frames.
    
    Inputs:
        sourcefilename (string): h5 filepath to get camera data from
        targetfilename (string): h5 filepath to save the new dataset to
        frame (array[bool]): boolean array representing the pixels to subset
        datarange tuple(int): section of frames to analyze (defaults to full file)
        normalize (bool): normalize each frame by sum if true (default true)'''
    
    im0 = getimage0(sourcefilename)
    
    sourcefile = h5py.File(sourcefilename, 'r')
    targetfile = h5py.File(targetfilename, 'w')
    
    datastart, datalength = datarange
    datalength = min(len(sourcefile['cameradata']['arrays']), datalength)
    datastart = max(datastart, 0)
    
    #This is a crude metric for scaling that sets the normalized value of the brightest pixel in the
    #first image to 240. This is used to ensure the normalized data can be saved as uint8 without
    #losing significant information. This might be part of what's causing the map volitility.
    
    #To do: find alternatives
    if normalize:
        scale = 240/np.max(im0/np.sum(im0))
    else:
        scale = 1
        
    #Used to pass the initial values into the multiprocessing function at each frame
    subsetframe_specific = functools.partial(subsetframe, frame=frame, normalize=normalize, normalfactor=scale)
    
    returnval=[]
    
    #Try-Except loop ensures everything closes properly if code fails
    try:
        #Initialize file
        targetfile.create_group("auxdata")
        targetfile.create_dataset("auxdata/subsetmap", data=frame)
        targetfile.create_dataset("auxdata/samplingrate", data=sourcefile['auxdata']['samplingrate'][()])
        targetfile.create_dataset("auxdata/normalizationscale", data=scale)
        im0_test = sourcefile['cameradata']['arrays'][0][frame]
        targetfile.create_group("cameradata")
        targetfile.create_dataset("cameradata/arrays", data=Parallel(n_jobs=ncores)(delayed(sectionsum_specific)(i) for i in sourcefile['cameradata']['arrays'][datastart:datalength]))
    finally:  
        sourcefile.close()
        targetfile.close()

def expand_fromsubset(vector, subsetmap):
    '''Given a 1D array of values, and a boolean array corresponding to a subset map, returns a
    map of the same size as the original image, where true elements are populated with the values
    and false elements are set to 0'''
    outp = np.zeros_like(subsetmap,dtype=vector.dtype)
    outp[subsetmap] = vector
    return outp

def h5_fft(h5filepath, datarange=(0,np.inf), sectionlength=8000,phasepixel=None, \
           usewindow=False, normalize = True, roi=None):
    '''Given a filepath to an h5 file with camera data, takes the fft of each
    individual pixel. It does this by splitting into smaller sections to ease
    processing strain; default is 10 seconds by convention.
    
    Inputs:
        h5filepath (string): h5 filepath to get camera data from
        datarange tuple(int): section of frames to analyze (defaults to full file)
        sectionlength (int): number of frames per section (default 8000)
        
    Output:
        tuple[array] with the fft frequencies and the fft data'''
    with h5py.File(h5filepath, 'r') as f:
        datastart, datalength = datarange
        datalength=min(len(f['cameradata']['arrays']), datalength)
        datastart = max(datastart,0)
        nsections = (datalength-datastart)//sectionlength
        imageshape = f['cameradata']['arrays'][0].shape
        dims = len(imageshape)
        
        if roi:
            roi = np.reshape(np.array(roi), (-1,2))
            roi = tuple([slice(roidim[0], roidim[1]) for roidim in roi])
        else:
            roi = tuple([slice(None)]*(dims))
        roi = tuple([slice(None)]) + roi
        
        
        if usewindow:
            win = scipy.signal.windows.tukey(sectionlength, 0.05)
            S_1 = np.sum(win)
            S_2 = np.sum(win**2)
            
        #calculate fft, then divide by phase of index pixel. 
        #Dividing by nsections allows us to get the average with a simple sum.
        dat = f['cameradata']['arrays'][datastart:(datastart+sectionlength)][roi].astype(np.float64)
        
        if normalize:
            dat /= np.sum(dat, axis=tuple(np.arange(1,len(imageshape)+1)))[newendaxes(dims)]
        if usewindow:
            dat -= np.mean(dat, axis=0)
            dat *= win[newendaxes(dims)]
        fft_transf=np.fft.rfft(dat, axis=0)*np.sqrt(2)/(sectionlength*nsections)
        del dat
        
        #index represents the xy coordinate of the pixel with greatest magnitude in frequency space at index 1.
        #it is arbitrarily chosen to normalize the phase
        if phasepixel is None:
            phasepixel = np.unravel_index(np.argmax(np.abs(fft_transf[1,:,:])), imageshape)
        index_slice = tuple([slice(None)]) + tuple(phasepixel)
        
        
        bigpixel_phase = fft_transf[index_slice]/np.abs(fft_transf[index_slice])
        fft_transf /= bigpixel_phase[newendaxes(dims)]
        
        #splits the data into sections, takes the fft of each, and averages.
        for i in range(1,nsections):
            dat = f['cameradata']['arrays'][datastart:(datastart+sectionlength)][roi].astype(np.float64)
            if normalize:
                dat /= np.sum(dat, axis=tuple(np.arange(1,len(imageshape)+1)))[newendaxes(dims)]
            if usewindow:
                dat -= np.mean(dat, axis=0)
                dat *= win[newendaxes(dims)]
                dat /= S_1
            else:
                dat /= sectionlength
                
            temp=np.fft.rfft(dat, axis=0)*np.sqrt(2)/(nsections)
            del dat
            
            bigpixel_phase = temp[index_slice]/np.abs(temp[index_slice])
            fft_transf += temp/bigpixel_phase[newendaxes(dims)]
            
        #calculates and returns numpy fft frequency conventions for sampling rate and section length
        samplingrate_transf = np.round(f['auxdata']['samplingrate'][()])
        freqs_transf = np.fft.rfftfreq(sectionlength, 1/samplingrate_transf)
    return freqs_transf, fft_transf

def isolate_frequency(full_fft, target_frequency, **kwargs):
    '''Uses h5_fft (or an existing fft data result) to return each pixels fft data
    at a given frequency.
    Inputs:
        target_frequency (int): frequency to return
        full_fft (tuple/array OR str): either the results from a previous full_fft run or a filepath to perform the fft on'''
    if type(full_fft) is str:
        full_fft = h5_fft(full_fft, **kwargs)
    target_index = findclosestoset(full_fft[0], [target_frequency])[0]
    return full_fft[1][target_index]

def singlefreq_fourier(h5filepath, frequency, hz=True, datalength=np.inf, normalized=True):
    '''Test to speed up calculating one frequency at a time instead of
    subsetting from the full fft. No significant speedups found, but it does save memory. 
    Likely areas for efficiency improvement if this becomes a part of future analysis.'''
    if hz: frequency = frequency*2*np.pi
    with h5py.File(h5filepath, 'r') as f:
        samplingrate = f['auxdata']['samplingrate'][()]
        deltatime = 1/samplingrate
        im0 = f['cameradata']['arrays'][0]
        index = np.unravel_index(np.argmax(im0), im0.shape)
        counter = np.zeros(im0.shape, dtype='complex128')
        datalength = min(datalength, len(f['cameradata']['arrays']))
        
        #Probably don't need to load entire file into memory
        for n, val in enumerate(f['cameradata']['arrays'][:datalength]):
            if normalized: counter += val*np.exp(-1j*deltatime*frequency*n)/np.sum(val)
            else: counter += val*np.exp(-1j*deltatime*frequency*n)
            
        bigpixel_phase = counter[index]/np.abs(counter[index])
    return counter/bigpixel_phase

def custom_cmap():
    '''Wrapper for a custom color map that matches the seismic color map, but has low and high values as the same color. Intended for plotting frequency maps.'''
    colors = [(0, 'blue'), (0.25, 'black'), (0.5, 'red'), (0.75, 'white'), (1, 'blue')]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)

def make_angleplot(fig, ax, arr, title, mask=None, **plotargs):
    '''Helper function for plotting the fft phase for a camera frame at a given frequency.    
    Inputs:
        fig, ax: existing matplotlib fig, ax
        arr (array): phases to plot
        title (string): title of graph
        mask (array[float]): alpha mask to overlay over image'''
    if mask is not None:
        color_map_overlay = ax.imshow(arr, cmap=custom_cmap(), vmin=-np.pi,vmax=np.pi,alpha=(mask/np.max(mask)), **plotargs)
    else:
        color_map_overlay = ax.imshow(arr, cmap=custom_cmap(), vmin=-np.pi,vmax=np.pi,alpha=1, **plotargs)
    cbar = fig.colorbar(color_map_overlay, cmap=custom_cmap, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
    cbar.ax.set_ylabel("complex phase")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

def save_angleplot(arr, title, filename, mask=None):
    '''Makes a graph as in make_angleplot and saves it to a given filepath
    Inputs:
        arr (array): phases to plot
        title (string): title of graph
        filename (string): filepath to save image to
        mask (array[float]): alpha mask to overlay over image'''
    fig, ax = plt.subplots()
    make_angleplot(fig, ax, arr, title, mask)
    plt.savefig(filename)
    plt.clf()
    
def make_scatterplot(fig, ax, values, title, xvals=None, ylim=None, **plotargs):
    '''Helper function to contain some plot arguments when making a scatterplot.
    Would not recommend using outside of other functions in this file'''
    if xvals is not None:
        indices = findclosestoset(values[1],xvals)
    else:
        indices = slice(None)
    ax.scatter(values[1][indices], values[0][indices], **plotargs)

    ax.tick_params(axis='both',which='both', bottom=False, left=False)

    ax.set_xscale('log')
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlim(1, 800)
    ax.grid(which='major', linestyle='--', linewidth='0.5', color='black', alpha=0.5)
    ax.set_title(title)
    
def save_scatterplot(values, title, filename, xvals=None, ylim=None, **plotargs):
    '''Saves scatterplot as made above. Would not recommend using outside of other functions in this file.'''
    fig, ax = plt.subplots()
    make_scatterplot(fig, ax, values, title, xvals=xvals, ylim=ylim, **plotargs)
    plt.savefig(filename)
    plt.clf()
    
def phase_to_mask(phasegrid, std=10):
    '''Given an array of phases, uses a gaussian blur and binarizes to turn into a mask with
    opposite phases having values of +-1.'''
    blurred = np.zeros(phasegrid.shape)
    
    #This creates a gaussian blurred frame representing absolute phase difference from 0
    phasegrid = np.abs(phasegrid)
    blurred = scipy.ndimage.gaussian_filter(phasegrid, std)
    
    #Binarizes the image
    blurred[np.where(blurred<np.pi/2)]=-1
    blurred[np.where(blurred>np.pi/2)]=1
    return blurred

def psd(series, samplingrate, sqrt = True, detrend='linear'):
    '''Calculates the psd of a series. 
    See (https://grattalab.com/elog/optlev/2023/10/25/conventions-for-spectra-spectral-densities-fft-normalization-etc/) for conventions.'''
    if detrend not in ['linear', 'mean', 'none', None]:
        raise ValueError("Valid options for detrend; linear, mean, none")
    if detrend == 'linear': series = lin_detrend(series)
    if detrend == 'mean': series = series-np.mean(series)
    
    spectrum = np.fft.rfft(series)
    freqs = np.fft.rfftfreq(len(series), d=1/samplingrate)
    p = 2/(samplingrate*len(series))*np.abs(spectrum)**2
    
    if sqrt: return np.sqrt(p), freqs
    else: return p, freqs

def windowed_psd(series, samplingrate, sqrt = True, detrend='linear', winsize=8000):
    '''Calculates the psd of a series, split into sections with a tukey window applied. 
    See (https://grattalab.com/elog/optlev/2023/10/25/conventions-for-spectra-spectral-densities-fft-normalization-etc/) for conventions.'''
    if detrend not in ['linear', 'mean', 'none', None]:
        raise ValueError("Valid options for detrend; linear, mean, none")
    if detrend == 'linear': series = lin_detrend(series)
    if detrend == 'mean': series = series-np.mean(series)
    
    #Getting values from window to properly scale psd
    sections = [series[i:i+winsize] for i in range(0, len(series), winsize)]
    win = scipy.signal.windows.tukey(winsize, 1)
    S_1 = np.sum(win)
    S_2 = np.sum(win**2)
    
    freqs = np.fft.rfftfreq(winsize, d=1/samplingrate)
    
    spectrum = np.array(np.fft.rfft(sections[0]*win))
    avgpsd = 2/(samplingrate*S_2)*np.abs(spectrum)**2
    avgpsd /= len(sections)
    
    for sec in sections[1:]:
        spectrum = np.array(np.fft.rfft(sec*win))
        p = 2/(samplingrate*S_2)*np.abs(spectrum)**2
        avgpsd += p/len(sections)
    
    if sqrt: return np.sqrt(avgpsd), freqs
    else: return avgpsd, freqs

def windowed_fft(series, samplingrate, detrend='linear', winsize=8000):
    '''Calculates the fft of a series, split into sections with a tukey window applied. 
    See (https://grattalab.com/elog/optlev/2023/10/25/conventions-for-spectra-spectral-densities-fft-normalization-etc/) for conventions.'''
    if detrend not in ['linear', 'mean', 'none', None]:
        raise ValueError("Valid options for detrend; linear, mean, none")
    if detrend == 'linear': series = lin_detrend(series)
    if detrend == 'mean': series = series-np.mean(series)
    
    sections = [series[i:i+winsize] for i in range(0, len(series), winsize)]
    win = scipy.signal.windows.tukey(winsize, 0.05)
    S_1 = np.sum(win)
    S_2 = np.sum(win**2)
    spec = []
    
    freqs = np.fft.rfftfreq(winsize, d=1/samplingrate)
    
    spectrum = np.array(np.fft.rfft(sections[0]*win))
    avgpsd = np.sqrt(2)/(S_1)*np.abs(spectrum)
    avgpsd /= len(sections)
    
    for sec in sections[1:]:
        spectrum = np.array(np.fft.rfft(sec*win))
        p = np.sqrt(2)/(S_1)*np.abs(spectrum)
        avgpsd += p/len(sections)
    return avgpsd, freqs


def findSectionSumsMasked(frame, mask, dims, normalize, roi):
    '''Parallel part of parallelSumsMasked'''
    if normalize:
        return np.squeeze(np.apply_over_axes(np.sum, (np.expand_dims(frame[roi], -1)/np.sum(frame[roi]))*mask, range(dims)))
    else:
        return np.squeeze(np.apply_over_axes(np.sum, (np.expand_dims(frame[roi], -1))*mask, range(dims)))

def parallelSumsMasked_h5(mask, h5filepath, datarange=(0,np.inf), dims=2, normalize=True, subset=None, roi=None):
    '''Given a weight map and a filepath to an h5 file with camera data, 
    returns the sum of each image weighted by the given map.
    
    Inputs:
        Mask: the mask to use when summing over the image
        h5filepath: filepath to camera data being summed over
        datarange tuple(int): section of frames to sum over (defaults to full file)
        dims: number of dimensions in each frame (default: 2)
            Note: this is 2 for unmodified camera data, and 1 for subset camera data'''
    if len(mask.shape) < dims or len(mask.shape) > dims+1: raise ValueError("Dimension and mask mismatch")
    if len(mask.shape) == dims: mask = np.expand_dims(mask,-1)
    nums = []
    
    datastart, datalength = datarange
    
    if roi:
        roi = np.reshape(np.array(roi), (-1,2))
        roi = tuple([slice(roidim[0], roidim[1]) for roidim in roi])
    else:
        roi = tuple([slice(None)]*(dims))
    

    sectionsum_specific = functools.partial(findSectionSumsMasked, mask=mask, dims=dims, normalize=normalize, roi=roi)
    
    f = h5py.File(h5filepath, 'r')
    datalength=min(len(f['cameradata']['arrays']), datalength)
    datastart=max(0,datastart)
    returnval=[]

    # ### parallel processing ###
    returnval = Parallel(n_jobs=ncores)(delayed(sectionsum_specific)(i) for i in f['cameradata']['arrays'][datastart:datalength])
    f.close()

    return returnval

#def fullfolder_analysis(mask, h5filepath, indices, dims=2, normalize=True):
    
        
def generate_masks(xfile, yfile, xfrequency, yfrequency, blurred=True):
    '''Given an x file, a y file, and a frequency for each, converts the
    phase at each pixel at each frequency to a map for weighted sums
    
    Output:
        The x and y maps as a numpy array stacked along the last axis'''
    
    xmap = isolate_frequency(xfile, xfrequency)
    xmap = xmap/np.abs(xmap)
    #phase_adjust = np.mean(np.abs(np.angle(xmap)))
    phase_adjust=0
    xmap = np.real(xmap*np.exp(-1j*phase_adjust))
    
    ymap = isolate_frequency(yfile, yfrequency)
    ymap = ymap/np.abs(ymap)
    #phase_adjust = np.mean(np.abs(np.angle(ymap)))
    phase_adjust=0
    ymap = np.real(ymap*np.exp(-1j*phase_adjust))
    
    
    if blurred:
        xmap = phase_to_mask(xmap)
        ymap = phase_to_mask(ymap)
    return np.dstack((xmap, ymap))

def manual_leftinv(matrix):
    '''Calculates left inverse using matrix multiplication (not SVD)'''
    return np.matmul(np.linalg.inv(np.matmul(matrix.T, matrix)), matrix.T)

def generate_quad_masks(shape):
    '''Given a 2x2 array shape, return a mask that mimics the qpd'''
    xquadmap = np.ones(shape)
    xquadmap[:, :shape[1]//2] = -1
    yquadmap = np.ones(shape)
    yquadmap[:shape[0]//2, :] = -1
    return np.dstack((xquadmap,yquadmap))

def invert_images(*images, real=True):
    '''Given a list of m images of the same shape and size n, turns them into an nxm matrix, and calculate the left inverse.
    Returns the matrices reshaped to the original image shape, dstacked.
    real=True takes the real part of a complex map first'''
    #Turns the maps into a nx2 matrix, then calculate the left inverse
    shape = images[0].shape
    maps = [image.flatten() for image in images]
    maps = np.squeeze(np.dstack(maps))
    if real:
        #phase_adjust = np.mean(np.abs(np.angle(maps)), axis=0)
        #maps = np.real(maps*np.exp(-1j*phase_adjust)[np.newaxis,:])
        maps = np.real(maps)
    maps_inv = manual_leftinv(maps)
    return maps_inv.T.reshape(shape+tuple([-1]))

def generate_diagonal_masks(xfile, yfile, xfrequency, yfrequency, real=True, xnormalized = True, ynormalized = True, **kwargs):
    '''Given an x file, a y file, and a frequency for each, generates maps from the diagonalization procedure
    shown here https://grattalab.com/elog/optlev/2024/04/25/applying-diagonalization-maps-to-11-27-camera-data/
    Inputs:
        xfile (string): filepath of h5 file with camera data for x data
        yfile (string): filepath of h5 file with camera data for y data
        xfrequency (int): frequency to diagonalize the x map at
        yfrequency (int): frequency to diagonalize the y map at
        real (bool): only keep the real part of the frequency response to generate the maps (default True)
        xnormalized, ynormalized (bool): set to true if the x and y files were normalized during subsetting (default true)
        kwargs passed to h5_fft'''
    shape = getimage0(xfile).shape
    if shape != getimage0(yfile).shape:
        raise ValueError("X File and Y file aren't the same shape")
        
    #Get x and y frequency responses at target frequencies
    x1 = isolate_frequency(xfile, xfrequency, **kwargs)
    #x1 = singlefreq_fourier(xfile, xfrequency)
    if xnormalized: x1 = x1 / getnormscale(xfile)
    y1 = isolate_frequency(yfile, yfrequency, **kwargs)
    #y1 = singlefreq_fourier(yfile, yfrequency)
    if ynormalized: y1 = y1 / getnormscale(yfile)
    
    #Turns the maps into a nx2 matrix, then calculate the left inverse
    return invert_images(x1,y1, real=real)

def calculate_snr(psd_vals, signal_values, maxval=None):
    '''Given results from a PSD and frequencies at which to measure the signal,
    return the ratio between the average signal power and the average non-signal power.
    Note: maxval is the maximum frequency to consider for noise'''
    if maxval is None:
        maxval = -1
    else:
        maxval = findclosestoset(psd_vals[1], [maxval])[0]
    signal_values = findclosestoset(psd_vals[1], signal_values)
    mask = np.full(len(psd_vals[1]), True)
    mask[signal_values] = False
    mask[maxval:] = False
    return np.mean(psd_vals[0][signal_values])/np.mean(psd_vals[0][mask])
        
def makeTransferFuncPlot(masks, xfile, yfile, zfile=None, xvals=None, ylim=None, datarange=(0,np.inf), filepath = None, normscale = [1,1], electrons=9, plotbase=None, dims=2, **plotargs):
    '''Plots a transfer function from x, y, and optionally z camera files. Optionally save the file to a given filepath
    If frequencies are specified, only those frequencies will be shown, and the transfer functions will be converted to
    force units based on those files
    Inputs:
        masks (array): x and y mask to test, stacked along last axis
        xfile (string): filepath of h5 file with camera data for x transfer data
        yfile (string): filepath of h5 file with camera data for y transfer data
        zfile (string): filepath of h5 file with camera data for z transfer data (default None)
        
        ylim: y limits of plots (default None)
        datarange tuple(int): section of frames to analyze from the camera datasets (defaults to full file)
        filepath (string): filepath to save save the image to (default None)
        dims (int): number of dimensions per image (default 2)
        
        xvals (array[int]): the frequencies to display and measure the force (default None, shows all frequencies)
        electrons (int): number of electrons for force normalization (default 9)
        
        plotbase (fig, array[ax]): pass existing transfer function plot fig/axs as input to plot multiple transfer functions on the same graph
        **plotargs: any additional elements to add to the scatterplot
    Outputs:
        (fig, axs): figure and axes for plot
        '''
    
    if zfile is not None:
        files = [xfile, yfile, zfile]
    else:
        files = [xfile,yfile]
    
    titles = ["x", "y", "z"]
    if plotbase is not None:
        fig, axs = plotbase
    else:
        fig, axs = plt.subplots(2,len(files), figsize=(18,3*len(files)), sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.175,wspace=0.1)
    
    for j in range(len(files)):
        #Gets x/y map weighted sums for each file 
        vals = parallelSumsMasked_h5(masks, files[j], datarange=datarange,dims=dims)
        samplingrate = getsamplingrate(files[j])
        for i in range(2):
            #Splits x/y measurements, calculates/plots psd
            data = windowed_psd(np.array(vals)[:,i]*normscale[i], samplingrate, winsize=int(samplingrate*10))
            title = f"Response in {titles[i]} to drive in {titles[j]}"    
            make_scatterplot(fig, axs[i,j], data, title, xvals=xvals, ylim=ylim, **plotargs)
    
    #Ensures y axis is properly labeled with force units or arbitrary
    if normscale == [1,1]:
        axs[0,0].set_ylabel(r'$\sqrt{S_x} [Arb/\sqrt{Hz}]$')
        axs[1,0].set_ylabel(r'$\sqrt{S_y} [Arb/\sqrt{Hz}]$')
    else:
        axs[0,0].set_ylabel(r'$\sqrt{S_x} [N/\sqrt{Hz}]$')
        axs[1,0].set_ylabel(r'$\sqrt{S_y} [N/\sqrt{Hz}]$')
    
    for j in range(len(files)):
        axs[-1,j].set_xlabel(r'$Freq [Hz]$')
        
    if filepath is not None:
        plt.savefig(filepath)
    return (fig, axs)

def force_calibration(masks, xfile, yfile, xbeadfile, ybeadfile, electrons=9, xvals=np.arange(1,100), datalength=np.inf, dims=2):
    '''Given x/y masks to test, files for camera data and bead data corresponding to x and y transfer functions,
    return the coefficient needed to convert to force units.
    Inputs:
        masks (array): x and y mask to test, stacked along last axis
        xfile (string): filepath of h5 file with camera data for x transfer data
        yfile (string): filepath of h5 file with camera data for y transfer data
        xbeadfile (string): filepath of h5 file with bead data for x transfer data
        ybeadfile (string): filepath of h5 file with bead data for y transfer data
        electrons (int): number of electrons on bead (default 9)
        xvals (array[int]): the frequencies of the comb at which to measure the force
        datalength (int): maximum number of frames to keep from the camera dataset (default infinity)
        dims (int): number of dimensions per image (default 2)'''
    files = [xfile, yfile]
    calib = [0,0]
    
    #Getting force from bead datafiles
    force = get_driveforce(xbeadfile, ybeadfile, electrons=electrons, xvals = xvals)
    
    #Calculates force from camera data as mean fft over all tested frequencies
    for i in range(2):
        vals = parallelSumsMasked_h5(masks, files[i], datalength=datalength, dims=dims)
        samplingrate = getsamplingrate(files[i])
        data = windowed_psd(np.array(vals)[:,i], samplingrate, winsize=int(samplingrate*10)) 
        indices = findclosestoset(data[1], xvals)
        calib[i] = force[i]/np.mean(data[0][indices])
    return calib, data

def get_driveforce(xfile, yfile, electrons=9, xvals = np.arange(1,100)):
    '''Given bead data files for X and Y drive, a list of comb frequencies,
    and the number of electrons, calculates the mean force applied over all
    target frequencies.
    '''
    
    xt_electrodes = BeadDataFile(xfile)
    yt_electrodes = BeadDataFile(yfile)
    xdrive_efield = (xt_electrodes.electrode_data[0]-xt_electrodes.electrode_data[1])*100*0.66/8.6e-3
    ydrive_efield = (yt_electrodes.electrode_data[0]-yt_electrodes.electrode_data[1])*100*0.66/8.6e-3

    xdrive_force = xdrive_efield*scipy.constants.e*electrons
    ydrive_force = ydrive_efield*scipy.constants.e*electrons

    #Using 1 seconds at QPD sampling rate of 5 kHz
    xforce_amp = windowed_psd(xdrive_force,5000,winsize=5000, detrend='none')
    yforce_amp = windowed_psd(ydrive_force,5000,winsize=5000, detrend='none')
    
    indices = findclosestoset(xforce_amp[1], xvals)
    
    xforce_avg = np.mean(xforce_amp[0][indices])
    yforce_avg = np.mean(yforce_amp[0][indices])
    return (xforce_avg, yforce_avg)

def findSectionSums(frame, gridsize, expandfrom1D, expandmap):
    '''See parallelSums'''
    imageshape = frame.shape
    
    if expandfrom1D:
        frame = expand_fromsubset(frame, expandmap)
        imageshape = expandmap.shape

    #making the edges for the grid on the image
    xleft = [imageshape[1]//gridsize[1] * i for i in range(gridsize[1])]
    xright = [imageshape[1]//gridsize[1] * i for i in range(1,gridsize[1])]
    xright.append(-1)
    ytop = [imageshape[0]//gridsize[0] * i for i in range(gridsize[0])]
    ybottom = [imageshape[0]//gridsize[0] * i for i in range(1,gridsize[0])]
    ybottom.append(-1)

    #computing the sum of each section and returning a list
    sumlist = [np.sum(np.sum(frame[ytop[j]:ybottom[j],xleft[i]:xright[i]])) for j in range(gridsize[0]) for i in range(gridsize[1])]
    return np.reshape(np.array(sumlist), (gridsize[0], gridsize[1]))

def parallelSectionSums(h5filepath,gridsize, expandfrom1D = False,datalength=-1):
    '''Given a camera dataset, this function splits it into a grid of sections and computes the sum
    of all pixels in each section for each frame. Returns a list with an element for each frame, with
    each element being an '''
    f = h5py.File(h5filepath, 'r')
    if datalength==-1: datalength=len(f['cameradata']['arrays'])
    returnval=[]
    
    
    if expandfrom1D: 
        expandmap = f['auxdata']['subsetmap'][()]
    else:
        expandmap = None
    
    sectionsums_specific = functools.partial(findSectionSums, gridsize=gridsize, expandfrom1D=expandfrom1D, expandmap=expandmap)
    
    # ### parallel processing ###
    returnval = Parallel(n_jobs=ncores)(delayed(sectionsums_specific)(i) for i in f['cameradata']['arrays'][:datalength])
    f.close()
    
    return returnval

def parallelSectionSums_file(h5filepath_in,gridsize,h5filepath_out,expandfrom1D=False,datalength=-1):
    '''Performs parallelSums on the h5 file h5filepath_in, then saves the results to the h5 file
    h5filepath_out'''
    sourcefile = h5py.File(h5filepath_in, 'r')
    targetfile = h5py.File(h5filepath_out, 'w')
    targetfile.create_group("auxdata")
    targetfile.create_dataset("auxdata/samplingrate", data=sourcefile['auxdata']['samplingrate'][()])
    sourcefile.close()
    targetfile.create_group("cameradata")
    targetfile.create_dataset("cameradata/arrays", data=parallelSectionSums(h5filepath_in, gridsize, datalength=datalength,expandfrom1D=expandfrom1D))
    
    
    
'''This class is intended for use combining multiple camera datasets into a single gravity dataset.
Hopefully to be outmoded by the new camera, which will make it legacy'''
class gravity_cameradataset():
    def __init__(self, directorypaths):
        if type(directorypaths) == str: self.directorypaths = [directorypaths]
        else: self.directorypaths = directorypaths
        self.__getfilepaths()
        self.__getsectionindices()
        
    '''The class acts as a list: if you pass the index of a shaking dataset to it, you can see 
    if the corresponding dataset was measured in its entirety by the camera, and determine
    the frame numbers and camera filepath for the corresponding data.'''
    def __contains__(self, key: tuple[int,int]):
        return key[1] in self.datasections[key[0]] or key[1] in self.calibsections[key[0]]
    def __getitem__(self, key: tuple[int,int]):
        dirnum, datasetnum = key
        if datasetnum in self.datasections[dirnum]:
            dirpath, secstart, secend = self.dataindices[dirnum][self.datasections[dirnum].index(datasetnum)]
            return "Data", dirpath, secstart, secend
        elif datasetnum in self.calibsections[dirnum]:
            dirpath, secstart, secend = self.calibindices[dirnum][self.calibsections[dirnum].index(datasetnum)]
            return "Calib", dirpath, secstart, secend
        else:
            raise ValueError("Likely an invalid dataset number: check the datasections/calibsections \
                                variables for valid dataset numbers")
            
    def getdata(self, key: tuple[int,int]):
        '''Returns the camera data corresponding to a specific segment'''
        datatype, dirpath, startind, endind = self[key]
        with h5py.File(dirpath) as f:
            return datatype, f['cameradata']['arrays'][startind:endind]
        
    def analyzedata_fullfft(self, verbose=False, takepsd=False, specsections = None, **kwargs):
        '''Given a mask, collect the fft of the sum weighted by that mask for all gravity data in the dataset.
        Setting takepsd to True averages the psds: otherwise, the complex ffts are averaged.
        Any additional keyword arguments are passed to parallelSumsMasked_h5.'''
        averagedfft = 0
        sectionsanalyzed = 0

        #Assuming samplingrate is 800 fps
        samplingrate = 800
        length = 8000
        window = scipy.signal.windows.tukey(length, 0.05)
        S_1 = np.sum(window)
        S_2 = np.sum(window**2)
        windowsettings = (S_1,S_2)
        freqs = np.fft.rfftfreq(8000, d=1/800)


        
        if not specsections:
            specsections = self.dataindices
        else:
            specsections = [[self[(i,j)][1:] for j in specsections[i]] for i in range(len(specsections))]
        
        for i, dirpath in enumerate(self.directorypaths):
            if verbose: print(f"Now analyzing directory {i}")
            for ind in tqdm(specsections[i], disable = not verbose):
                # Each index directs to the correct frames of the right camera data file
                dirpath = os.path.join(dirpath,ind[0])
                edges = (ind[1],ind[2])

                # Take the sums over the camera frames, multiply by the tukey window
                _, curfft = h5_fft(dirpath, datarange=edges, **kwargs)
                # Take the psd before averaging if desired
                if takepsd:
                    averagedfft += np.abs(curfft) * np.sqrt(2/(S_2*samplingrate))
                else:
                    averagedfft += curfft
                    
            # Divide to rescale the average properly
            sectionsanalyzed += len(specsections[i])

        return averagedfft/sectionsanalyzed
        
    def analyzedata(self, mask, verbose=False, takepsd=False, specsections = None, **kwargs):
        '''Given a mask, collect the fft of the sum weighted by that mask for all gravity data in the dataset.
        Setting takepsd to True averages the psds: otherwise, the complex ffts are averaged.
        Any additional keyword arguments are passed to parallelSumsMasked_h5.'''
        averagedfft = 0
        sectionsanalyzed = 0

        #Assuming samplingrate is 800 fps
        samplingrate = 800
        length = 8000
        window = scipy.signal.windows.tukey(length, 0.05)
        S_1 = np.sum(window)
        S_2 = np.sum(window**2)
        windowsettings = (S_1,S_2)
        freqs = np.fft.rfftfreq(8000, d=1/800)
        
        if not specsections:
            specsections = self.dataindices
        else:
            specsections = [[self[(i,j)][1:] for j in specsections[i]] for i in range(len(specsections))]
        
        for i, dirpath in enumerate(self.directorypaths):
            if verbose: print(f"Now analyzing directory {i}")
            for ind in tqdm(specsections[i], disable = not verbose):
                # Each index directs to the correct frames of the right camera data file
                dirpath = os.path.join(dirpath,ind[0])
                edges = (ind[1],ind[2])

                # Take the sums over the camera frames, multiply by the tukey window
                framesums = parallelSumsMasked_h5(mask, dirpath, \
                                                          datarange=edges, **kwargs)
                framesums -= np.mean(framesums, axis=0)
                framesums *= window[:,np.newaxis]
                # Take the psd before averaging if desired
                if takepsd:
                    averagedfft += np.abs(np.fft.rfft(framesums, axis=0)) * np.sqrt(2/(S_2*samplingrate))
                else:
                    averagedfft += np.fft.rfft(framesums, axis=0)
                    
            # Divide to rescale the average properly
            sectionsanalyzed += len(specsections[i])

        return averagedfft/sectionsanalyzed
    
    def analyze_HQPDdata(self, beaddirectorypaths, verbose=False, takepsd=False):
        '''Given a list of directory paths corresponding to the HQPD data, 
        calculate the averaged fft for the HQPD data of all data sections measured by the camera.
        Setting takepsd to True averages the psds: otherwise, the complex ffts are averaged.'''
        self.beaddirectorypaths = beaddirectorypaths
        averagedfft = 0
        sectionsanalyzed = 0

        samplingrate = 5000
        length = 50000
        window = scipy.signal.windows.tukey(50000, 0.05)
        S_1 = np.sum(window)
        S_2 = np.sum(window**2)
        windowsettings = (S_1,S_2)
        freqs = np.fft.rfftfreq(50000, d=1/5000)
        
        if not specsections:
            specsections = self.dataindices
        
        for i, dirpath in enumerate(self.directorypaths):
            if verbose: print(f"Now analyzing directory {i}")
            for ind in tqdm(specsections[i], disable = not verbose):
                # Extract the x and y HQPD data from the correct bead directory paths
                bdf = BeadDataFile(os.path.join(beaddirectorypaths[i],f"shaking_{ind}.h5"))
                bdfdata = np.dstack([bdf.x2,bdf.y2]).squeeze()
                bdfdata -= np.mean(bdfdata, axis=0)
                bdfdata *= window_beaddatafile[:,np.newaxis]
                if takepsd:
                    averagedfft += np.abs(np.fft.rfft(bdfdata, axis=0)) * np.sqrt(2/(S_2*samplingrate))
                else:
                    averagedfft += np.fft.rfft(bdfdata, axis=0)
                    
            sectionsanalyzed += len(specsections[i])

        return averagedfft/sectionsanalyzed
        
    def __getfilepaths(self):
        '''Private function to sort filepaths in each directory'''
        self.subfilepaths = []
        for k, directorypath in enumerate(self.directorypaths):
            filepaths = os.listdir(directorypath)
            filepaths.sort(key=lambda f: int(re.match(r'segment(\d+).h5', f).group(1)))
            self.subfilepaths.append(filepaths)
    
    def __getsectionindices(self):
        '''Private function to sort out the locations of each DAC dataset in the camera data,
        thereby tracking which datasets are usable for camera analysis and which are calibration datasets'''
        self.dataindices = [[] for i in range(len(self.directorypaths))]
        self.datasections = [[] for i in range(len(self.directorypaths))]
        self.calibindices = [[] for i in range(len(self.directorypaths))]
        self.calibsections = [[] for i in range(len(self.directorypaths))]
        self.missedsections = [[] for i in range(len(self.directorypaths))]
        for k, directorypath in enumerate(self.directorypaths):
            filepaths = self.subfilepaths[k]

            # NOTE: nchunks represents the number of section endings seen so far. This means each dataset starts
            # with data from the nchunks+1th section (chunk refers to DAC dataset, indexed from 0)
            lastdt=0
            nchunks = -1  # It seems weird to start at -1, but this is needed to make the indices work out
            lastts=0
            missedsections=[]
            print(f"Now loading {directorypath}.")
            for l, fp in enumerate(filepaths):
                fullpath = os.path.join(directorypath,fp)
                # Uses skips in the timestamp data due to DAC processing to find the end frames for each section
                with h5py.File(fullpath) as f:
                    dts = np.diff(f['cameradata']['timestamps'][:])
                    lastts = f['cameradata']['timestamps'][-1]
                sectionends = np.where(dts>1e8)[0]

                # If the length of the section spanning from the end of the previous dataset
                # to the start of the current dataset is longer than 10 seconds, then a section end
                # must have been missed during camera downtime. This accounts for the missing end
                if sectionends[0]+lastdt > 8000:
                    if sectionends[0] != 7999:
                        nchunks += 1
                    self.missedsections[k].append(nchunks)

                # Ensures the first and last sections in a file are counted only if they are complete.
                # ie, if the camera starts recording frame 0 exactly when the DAC starts shaking
                if sectionends[0] == 7999: 
                    sectionends = np.insert(sectionends, 0, -1)
                    sectionends = np.append(sectionends, 56000)

                lastdt = 56000-sectionends[-1]

                # Get the edges of each fully recorded dataset, and save them to an array
                for i in range(len(sectionends)-1):
                    goodindex = (fullpath, sectionends[i] + 1,  sectionends[i+1] + 1)
                    if (i + 1 + nchunks) % 10 == 9: #This is 9 since the end of section 9 is the start of section 10
                        self.calibindices[k].append(goodindex)
                        self.calibsections[k].append(nchunks+1+i)
                    else:
                        self.dataindices[k].append(goodindex)
                        self.datasections[k].append(nchunks+1+i)
                nchunks += len(sectionends)