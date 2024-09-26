'''-----------------------------------------------------------------------------
                                mod_utility.py

 Description: General python utilities

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import json
import math
import logging
import warnings
import itertools
import numpy as np
from shutil import move
from tempfile import mkstemp
from os import fdopen, remove
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
from pathlib import Path

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Externally installed modules
# None

# Local imports
from matkit.core import mod_math

'''-----------------------------------------------------------------------------
                              MODULE VARIABLES
-----------------------------------------------------------------------------'''
module_name = "mod_utility.py"

EPSILON = sys.float_info.epsilon  # Smallest possible difference.

# Multilevel logger
class MultilineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        save_msg = record.msg
        output = ""
        for line in save_msg.splitlines():
            record.msg = line
            output += super().format(record) + "\n"
        record.msg = save_msg
        record.message = output
        return output

class Plot_Settings:

    xlabel = None
    ylabel = None
    zlabel = None

    xlimits = None
    ylimits = None
    zlimits = None

    save_filename = None

    label_fontsize = None

    is_console_plot = False

    def __init__(self, *args, **kwargs):

        # Read input class members
        self.xlabel = kwargs.get('xlabel', 'x')
        self.ylabel = kwargs.get('ylabel', 'y')
        self.zlabel = kwargs.get('zlabel', 'z')

        self.xlimits = kwargs.get('xlimits', None)
        self.ylimits = kwargs.get('ylimits', None)
        self.zlimits = kwargs.get('zlimits', None)

        self.label_fontsize = kwargs.get('label_fontsize', 14)

        self.save_filename = kwargs.get('save_filename', None)
        self.is_console_plot = kwargs.get('is_console_plot', False)


'''----------------------------------------------------------------------------
                                 SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################


def get_degree_of_best_polyfit(x, y, degree_list=None, max_degree=None, is_even_degree=False, plot_dir=None, plot_filename=None, plot_title=None):

    # Sanity check
    if len(x) != len(y):
        sys.stderr.write("Error: In module 'mod_utility'\n")
        sys.stderr.write("       In subroutine 'get_degree_of_best_polyfit'\n")
        sys.stderr.write("       Length of input arrays 'X':%d and 'Y':%d differ\n" %(len(x), len(y)))
        sys.stderr.write("       Terminating!!!\n")
        exit(1) 

    if degree_list is None:
    
        if max_degree is None:
            max_degree = len(x)-1
        elif max_degree > len(x)-1:
            sys.stderr.write("Error: In module 'mod_utility'\n")
            sys.stderr.write("       In subroutine 'get_degree_of_best_polyfit'\n")
            sys.stderr.write("       The input value 'max_degree': %d should at most be 'len(X)-1':%d\n" %(max_degree, len(x)-1))
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        if is_even_degree:
            degree_list = np.arange(2, max_degree+1,2)
        else:
            degree_list = np.arange(2, max_degree+1,1)

    else:
        if np.max(degree_list) > len(x)-1:
            sys.stderr.write("Error: In module 'mod_utility'\n")
            sys.stderr.write("       In subroutine 'get_degree_of_best_polyfit'\n")
            sys.stderr.write("       The value of maximum degree in 'degree_list' should at most be 'len(X)-1'\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1) 
            
    min_rms_error = 1e10 # Set to a large value
    best_degree = 2 # atleast a 2nd degree polynoimal is needed
    rms_error_list = []
    
    X = np.copy(x).reshape((len(x),1))
    Y = np.copy(y).reshape((len(y),1))

    for degree in degree_list:
       
        # Train the polynomial with the data
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)

        # Linear regression
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, Y)
        
        # Compare self (training) error as we do not split data into test and train
        X_poly_test = poly_features.fit_transform(X)
        Y_poly_predict = poly_reg.predict(X_poly_test)
        poly_rms_error = np.sqrt( mean_squared_error(Y, Y_poly_predict) )
        rms_error_list.append(poly_rms_error)

        # Find minimum error
        if min_rms_error > poly_rms_error:
            min_rms_error = poly_rms_error
            best_degree = degree

    if plot_filename is not None:
        
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(6.0, 4.0))
        ax = fig.add_subplot(frame_on=True)
    
        plt.plot(degree_list, rms_error_list)
        ax.set_yscale('log')
        ax.set_xlabel('Degree')
        ax.set_ylabel('RMSE')
        
        plt.title(plot_title)
        plt.savefig(plot_dir + '/' + plot_filename + '.pdf', bbox_inches='tight', pad_inches=0.02)
        plt.close()

    return best_degree

##################
### SUBROUTINE ###
##################


def get_higher_precedence(precedence_arr, val, abs_tol):

    '''
    Get all entities with higher precede above me (including me)
    Input argument: precedence_arr contains entities with highest to lowest precedence
    '''
    # Output arr
    higher_precedence_arr = np.empty(0)

    if val is None:
        highter_precedence_arr = np.array(precedence_arr)
    else:
        # Loop from higheest to lowest precedence, locate yourself and break on location
        for precede in precedence_arr:
            higher_precedence_arr = np.append(higher_precedence_arr, precede)
            if math.isclose(val, precede, abs_tol=abs_tol):
                break

    return higher_precedence_arr


##################
### SUBROUTINE ###
##################


def ribbon_plot(x_grid, y_grid, data_2d_arr,
                marker_pos_2d_list=None, marker_size=10,
                x_limits=None, y_limits=None, z_limits=None,
                x_label=None, y_label=None, z_label=None, title=None,
                n_x_ticks=10, n_y_ticks = 10, n_z_ticks=10,
                ribbon_width=0.1):

    '''
    Each row of data_2d_arr is a ribbon data, whose y placemnet is determined
    by the correspsonding y grid value of the row index. Each row of dtata_2d_arr
    spans x_grid.

    Optional markers can be added along each ribbon. marker_pos_2d_list is a
    list of arrays. Each entity of the list is an array and can be of different
    dimension for each ribbon.

    returns matplotlib figure.
    '''

    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    #--------------#
    # Setup figure #
    #--------------#
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Y = np.linspace(-ribbon_width/2, ribbon_width/2, 2)
    X, Y = np.meshgrid(x_grid, Y)

    #----------------------------------#
    # Plot each ribbon with the offset #
    #----------------------------------#
    for idx, (y_offset, ribbon_data) in enumerate(zip(y_grid, data_2d_arr)):

        if marker_pos_2d_list is not None:
            marker_pos_x_arr = marker_pos_2d_list[idx]
            n_markers = len(marker_pos_x_arr)
            if n_markers > 0:
                marker_pos_y_arr = y_offset * np.ones(n_markers)
                marker_pos_z_arr = np.zeros(n_markers)

                ax.scatter(marker_pos_x_arr, marker_pos_y_arr, marker_pos_z_arr,
                           color='r', alpha=0.9, marker = 'o', s = marker_size)

        # Plot the ribbon surface.
        surf = ax.plot_surface(X, Y+y_offset, np.array([ribbon_data, ribbon_data]), cmap=cm.rainbow,
                               linewidth=0, antialiased=False)

    #----------------------------#
    # Customize axes if required #
    #----------------------------#
    if x_limits is not None:
        ax.set_xlim(x_limits)

    if y_limits is not None:
        ax.set_ylim(y_limits)

    if z_limits is not None:
        ax.set_zlim(z_limits)

    ax.zaxis.set_major_locator(LinearLocator(n_z_ticks))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.xaxis.set_major_locator(LinearLocator(n_x_ticks))


    #-----------------------------#
    # Set axes labels if required #
    #-----------------------------#
    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if z_label is not None:
        ax.set_zlabel(z_label)

    if title is not None:
        ax.set_title(title)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=20)

    return fig

##################
### SUBROUTINE ###
##################


def test_ribbon_plot():

    import numpy as np
    import matplotlib.pyplot as plt

    x_grid = np.linspace(-10.0, 10.0, 21)
    y_grid = np.linspace(-5.0, 5.0, 11)

    data_2d_arr = np.zeros((len(y_grid), len(x_grid)))

    marker_pos_2d_list = np.zeros((len(y_grid), 5))

    for i in range(0, len(y_grid)):
        data_2d_arr[i] = np.sqrt(x_grid**2)
        marker_pos_2d_list[i] = np.array([1, 2, 3, 4, 5])

    fig = ribbon_plot(x_grid, y_grid, data_2d_arr, marker_pos_2d_list)
    plt.show()

##################
### SUBROUTINE ###
##################

def digitize_data_to_stacked_hist(value_2d_np_arr, n_bins):

    # Find the maximum and minimum value
    val_min = value_2d_np_arr.min()
    val_max = value_2d_np_arr.max()

    # Create bin values
    bins = np.linspace(val_min, val_max, n_bins)

    # Bin the values, digitize
    binned_2d_arr = np.empty((0, n_bins))
    for value_arr in value_2d_np_arr:
        digitized = np.digitize(value_arr, bins)
        binwise_count_arr = np.zeros(n_bins)
        for idx in digitized:
            binwise_count_arr[idx-1] = binwise_count_arr[idx-1] + 1
        binwise_count_arr = binwise_count_arr/np.sum(binwise_count_arr)
        binned_2d_arr = np.append(binned_2d_arr, np.array([binwise_count_arr]), axis=0)

    return (bins, binned_2d_arr)

##################
### SUBROUTINE ###
##################

def find_nearest_idx(array, value):

    '''
    Find nearest index
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

##################
### SUBROUTINE ###
##################


def get_equispaced_entities_from_distribution(entity_value_arr, n_equispaced):

    '''
    Given a distribution of values --> entity_value_arr (indices are the
    entity ids) returns atmost n_equispaced entity ids from distribution of
    values. The idea is that chosen entity indices will have entity values as
    far apart as possible.
    '''

    # Find the maximum and minimum value
    val_min = np.amin(entity_value_arr)
    val_max = np.amax(entity_value_arr)

    # Create bin values as many as requested n_equispaced values
    n_bin = n_equispaced
    bins = np.linspace(val_min, val_max, n_bin+1)

    # Infinitesmally increase the last bin to the right, so that the highest is
    # not excluded. This is because of numpy digitization. This can be remedied
    # if we use right=True option, but in that case the first bin has to be
    # infinitesmally extended to the left to include the lowest elemnt into the
    # first bin
    bins[n_bin] = bins[n_bin] + 0.0001

    # size of entities is same as size of digitized
    n_entities = len(entity_value_arr)

    # Bin the values
    digitized = np.digitize(entity_value_arr, bins)

    # Get list of entity indices in a given bin
    def get_binwise_entity_ids(bidx):
        binwise_entity_ids = []
        for eidx in range(0,n_entities):
            loc_bidx = digitized[eidx] - 1 # Bin index, digitized starts from 1
            if loc_bidx == bidx:
                binwise_entity_ids.append(eidx)
        return binwise_entity_ids

    # Binwise count of entities
    binwise_count_arr = np.zeros(n_bin)
    for bidx in range(0, n_bin):
        binwise_count_arr[bidx] = len(get_binwise_entity_ids(bidx))

    # Select equispaced values, NOTE: all bins need not be populated.
    selected_entity_ids = []
    for bidx in range(0, n_bin):

        # Get entity indices belonging to the current bin
        bin_entity_ids = get_binwise_entity_ids(bidx)

        # Is the bin populated, else go to next
        if len(bin_entity_ids) == 0:
            continue
        else:
            bin_entity_value_arr = entity_value_arr[bin_entity_ids]

        # For the first bin, choose index with lowest entity value
        if bidx == 0:
            loc_minval_entity_idx = np.argmin(bin_entity_value_arr)
            # Get corresponding global entity idx
            selected_entity_idx = bin_entity_ids[loc_minval_entity_idx]
            
        # For the last bin choose index with highest entity value
        elif bidx == n_bin-1:
            loc_maxval_entity_idx = np.argmax(bin_entity_value_arr)
            # Get corresponding global entity idx
            selected_entity_idx = bin_entity_ids[loc_maxval_entity_idx]   

        # For intermediate bins, choose index that is closest to mean bin value
        else:
            bin_mean_value = 0.5 * (bins[bidx] + bins[bidx+1])
            loc_meanval_entity_idx = find_nearest_idx(array=bin_entity_value_arr, value=bin_mean_value)
            # Get corresponding global entity idx
            selected_entity_idx = bin_entity_ids[loc_meanval_entity_idx]  

        # Push entity idx
        selected_entity_ids.append(selected_entity_idx)

    return selected_entity_ids

##################
### SUBROUTINE ###
##################

def test_get_equispaced_entities_from_distribution( ):

    # Generate values drawn from normal distribution
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)

    actual_mean = abs(mu - np.mean(s))
    actual_sd = abs(sigma - np.std(s, ddof=1))

    # Plot distribution
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')

    # Generate equispaced points in the distribution
    n_equispaced = 10
    equispaced_ids = get_equispaced_entities_from_distribution(entity_value_arr=s, n_equispaced=n_equispaced)

    print(np.amin(s))
    print(np.amax(s))
    print(s[equispaced_ids])

    # plot the points as orange dots. Note: we might not get exactly n_equispaced points, because some bins may be empty in a general distibution
    for idx in equispaced_ids:
        plt.plot([s[idx]], [0], marker='v', markersize=6, color='orange')

    plt.show()

##################
### SUBROUTINE ###
##################


def create_logger(
        log_object_name=None, log_file_name=None, is_log_console=True,
        file_log_level="DEBUG", console_log_level="ERROR"):
    '''
    Logging levels: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
    '''

    if log_object_name is None:
        sys.stderr.write("Error: In module 'mod_utility'\n")
        sys.stderr.write("       In subroutine 'create_logger'\n")
        sys.stderr.write("       Input variable 'log_object_name' is " \
                         "required\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Create logger with object name log_object_name
    logger = logging.getLogger(log_object_name)
    # The root logger defaults for WARNING level. So set it to DEBUG level.
    # Latter settings of higher level can overwrite this.
    logger.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = MultilineFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler, add formatter to it and add handler to logger
    if log_file_name is not None:
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(getattr(logging, file_log_level))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Create console handler, add formatter to it and add handler to logger
    if is_log_console:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_log_level))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

##################
### SUBROUTINE ###
##################


def join_list(a, sep=" "):

    return sep.join(str(x) for x in a)

##################
### SUBROUTINE ###
##################

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

##################
### SUBROUTINE ###
##################


def copyfiles_starred(src_dir, dest_dir, starred_search):

    import glob, os, shutil

    files = glob.iglob(os.path.join(src_dir, starred_search))
    for fl in files:
        if os.path.isfile(fl):
            shutil.copy2(fl, dest_dir)

##################
### SUBROUTINE ###
##################


def convert_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%d:%02d:%02d" % (hours, minutes, seconds)

##################
### SUBROUTINE ###
##################


def error_check_path_exists(pathname, module, subroutine):

    if not os.path.exists(pathname):
        sys.stderr.write("Error: In module '%s'\n" % (module))
        sys.stderr.write("       In subroutine '%s'\n" % (subroutine))
        sys.stderr.write("       '%s' does not exist\n" % (pathname))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def error_check_dir_exists(dirname, module, subroutine):

    if not os.path.isdir(dirname):
        sys.stderr.write("Error: In module '%s'\n" % (module))
        sys.stderr.write("       In subroutine '%s'\n" % (subroutine))
        sys.stderr.write("       Directory '%s' does not exist\n" % (dirname))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def error_check_file_exists(filename, module, subroutine):

    if not os.path.isfile(filename):
        sys.stderr.write("Error: In module '%s'\n" % (module))
        sys.stderr.write("       In subroutine '%s'\n" % (subroutine))
        sys.stderr.write("       File '%s' does not exist\n" % (filename))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def error_check_argument_required(arg_val, arg_name, module, subroutine,
                                  valid_args=None):

    if arg_val is None:
        sys.stderr.write("Error: In module '%s'\n" % (module))
        sys.stderr.write("       In subroutine '%s'\n" % (subroutine))
        sys.stderr.write("       Input argument '%s' is required\n" % (arg_name))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    if valid_args is not None:
        if arg_val not in valid_args:
            sys.stderr.write("Error: In module '%s'\n" % (module))
            sys.stderr.write("       In subroutine '%s'\n" % (subroutine))
            sys.stderr.write("       Invalid value '%s' passed for input "\
                             "argument '%s'\n" % (arg_val, arg_name))
            sys.stderr.write("       Allowed arguments: '%s'\n"
                             % (", ".join(map(str, valid_args))))
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

##################
### SUBROUTINE ###
##################


def replace(file_path, pattern, subst=None, replace_entire_line=False):
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if pattern in line:
                    if replace_entire_line:
                        if subst is not None:
                            new_file.write(subst + "\n")
                    else:
                        new_file.write(line.replace(pattern, subst))
                else:
                    new_file.write(line)
    # Remove original file
    remove(file_path)
    # Move new file
    move(abs_path, file_path)

##################
### SUBROUTINE ###
##################


def get_data(expr):
    # Grab data from file
    os.system("%s>eout" % (expr))
    fin = open("eout", 'r')
    line = fin.readline().split()
    return (float(line[0]))

##################
### SUBROUTINE ###
##################


def Silent_File_Remove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred

##################
### SUBROUTINE ###
##################


def read_array_data(flnm, n_headers=0, separator=None, data_type=None):

    # Check if the file exists
    if not os.path.isfile(flnm):
        sys.stderr.write("Error: In module 'mod_utility.py'\n")
        sys.stderr.write("       In subroutine 'read_float_data'\n")
        sys.stderr.write("       File '%s' does not exists\n" % (flnm))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    fh = open(flnm, 'r')
    data = []
    for i, line in enumerate(fh, start=0):
        if i >= n_headers:
            row = line.split(separator)
            data.append(row)
    fh.close()

    if data_type is None or data_type == 'string':
        return data
    elif data_type == 'float':
        float_data = []
        for row in data:
            float_row = [float(a) for a in row]
            float_data.append(float_row)
        return np.array(float_data)
    elif data_type == 'int':
        int_data = []
        for row in data:
            int_row = [int(a) for a in row]
            int_data.append(int_row)
        return np.array(int_data)
    else:
        sys.stderr.write("Error: In module 'mod_utility.py'\n")
        sys.stderr.write("       In subroutine 'read_float_data'\n")
        sys.stderr.write("       Unknown data type\n")
        sys.stderr.write("       Allowed data types: 'string', 'float'\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def get_waiting_dirs(root_dir=None, prefix=''):
    '''
    1. Sweeps the 'root_dir' and returns all directories with a #WAITING# or
       # PROCESSING# or #ABORTED# flag.
    2. If #ABORTED# flag, this subroutine renames it to #WAITING# flag. The
       assumption in doing so is that #ABORTED# flags are due to  voluntary
       stopping of the previous run.
    3. If #PROCESSING# flag, renames it to #WAITING# flag. The assumption is
       that #PROCESSING# flags are due to involuntary stopping of the previous
       run, possibly due to crossing the wall time limit.
    '''

    if root_dir is None:
        sys.stderr.write("Error: In module 'mod_utility.py'\n")
        sys.stderr.write("       In subroutine 'get_waiting_dirs'\n")
        sys.stderr.write(
            "       root_dir is needed to search for waiting jobs\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    wait_tag = "#" + prefix + "WAITING#"
    abort_tag = "#" + prefix + "ABORTED#"
    proc_tag = "#" + prefix + "PROCESSING#"

    all_dirs = [x[0] for x in os.walk(root_dir)]
    wait_list = []
    for each_dir in all_dirs:

        if os.path.isfile(each_dir + "/" + wait_tag):
            wait_list.append(each_dir)

        if os.path.isfile(each_dir + "/" + abort_tag):
            os.rename(each_dir  + "/" + abort_tag, each_dir + "/" + wait_tag)
            wait_list.append(each_dir)

        if os.path.isfile(each_dir + "/" + proc_tag):
            os.rename(each_dir + "/" + proc_tag, each_dir + "/" + wait_tag)
            wait_list.append(each_dir)

    return wait_list

##################
### SUBROUTINE ###
##################


def get_tagged_dirs(root_dir=None, tag_flnm=None):
    '''
    Sweeps the 'root_dir' and returns all directories with a tag_flnm file
    present in int
    '''

    if root_dir is None:
        sys.stderr.write("Error: In module 'mod_utility.py'\n")
        sys.stderr.write("       In subroutine 'get_tagged_dirs'\n")
        sys.stderr.write(
            "       root_dir is needed to search for tagged jobs\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    all_dirs = [x[0] for x in os.walk(root_dir)]
    tagged_list = []
    for each_dir in all_dirs:

        if os.path.isfile(each_dir + "/" + tag_flnm):
            tagged_list.append(each_dir)

    return tagged_list

##################
### SUBROUTINE ###
##################


def str_list_to_float_list(str_list):

    return [float(i) for i in str_list]

##################
### SUBROUTINE ###
##################


class json_numpy_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

##################
### SUBROUTINE ###
##################


def smooth_plot(x_arr, y_arr, order=3, n_points=300):


    from scipy.interpolate import make_interp_spline, BSpline

    x_arr_smooth = np.linspace(min(x_arr),max(x_arr),n_points) #300 represents number of points to make between T.min and T.max

    spl = make_interp_spline(x_arr, y_arr, k=order) #BSpline object
    y_arr_smooth = spl(x_arr_smooth)

    return (x_arr_smooth, y_arr_smooth)

##################
### SUBROUTINE ###
##################

def interpolate_2d_matrix_list_depthwise(matrix_list, param_arr, param, interpolate_kind='cubic'):

    # Determine the dimensions of the matrix arr
    n_depth = len(matrix_list)
    n_row = len(matrix_list[0])
    n_col = len(matrix_list[0][0])

    # Sanity check
    if len(param_arr) != n_depth:
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'interpolate_matrix_list_depthwise'\n")
        sys.stderr.write( "      Depth of matrix should match length of param_arr\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Store interpolations of each element for all depths
    interp_function_matrix = [[None] * n_col] * n_row
    for ridx in range(0, n_row):
        for cidx in range(0, n_col):
                element_arr = matrix_list[:, ridx, cidx]
                interp_element = interp1d(param_arr, element_arr, interpolate_kind)
                interp_function_matrix[i][j] = interp_element

    interp_matrix = np.zeros((n_row, n_col))
    for ridx in range(0, n_row):
        for cidx in range(0, n_col):
            interp_matrix[ridx][cidx] = interp_function_matrix[ridx][cidx](param)
    
    
    return interp_matrix

def interpolate_nd_array(nd_arr, param_arr, param, interpolate_kind='cubic', axis=0):

    interp_function = interp1d(param_arr, nd_arr, axis=axis)
    return interp_function(param)

##################
### SUBROUTINE ###
##################


def calculate_2d_envelops(x, y):

    from numpy import array, sign, zeros
    from scipy.interpolate import interp1d

    # Prepend the first value of (y) to the interpolating values. This forces
    # the model to use the same starting point for both the upper and lower
    # envelope models.
    u_x = [x[0],]
    u_y = [y[0],]

    l_x = [x[0],]
    l_y = [y[0],]

    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y
    # respectively.
    for k in range(1,len(y)-1):
        if (sign(y[k]-y[k-1])>=0) and (sign(y[k]-y[k+1])>=0):
            u_x.append(x[k])
            u_y.append(y[k])

        if (sign(y[k]-y[k-1])<=0) and (sign(y[k]-y[k+1])<=0):
            l_x.append(x[k])
            l_y.append(y[k])

    # Append the last value of (y) to the interpolating values. This forces the
    # model to use the same ending point for both the upper and lower envelope
    # models.
    u_x.append(x[-1])
    u_y.append(y[-1])

    l_x.append(x[-1])
    l_y.append(y[-1])

    # Return lower and upper envelops
    return (l_x, l_y, u_x, u_y)

##################
### SUBROUTINE ###
##################


def remove_duplicates_in_tagged_arrays(a, b):

    '''
    If a contains duplicates rmoves corresponding elements of both a and b
    '''
    
    ua, uia = np.unique(a, return_index=True)
    ub = np.empty(0)
    for idx in uia:
        ub = np.append(ub, b[idx])

    return(ua, ub)

##################
### SUBROUTINE ###
##################


def convert_2d_grid_ids_to_flattened_idx(w_idx, h_idx, width):

    '''
            y
            |
     height |  <----------
            |             |
            |  ---------->
            |_______________________x
            0             width
    '''

    return w_idx + (width * h_idx)

##################
### SUBROUTINE ###
##################


def convert_flattened_idx_to_2d_grid_ids(idx, width):

    w_idx = idx % width # Remainder
    h_idx = idx // width # Integer division

    return (w_idx, h_idx)

##################
### SUBROUTINE ###
##################


def convert_3d_grid_ids_to_flattened_idx(w_idx, h_idx, d_idx, width, height):

    '''
            y
            |
     height |  <----------           "depth into the page"
            |             |
            |  ---------->
            |_______________________x
            0             width
    '''

    return w_idx + (width * h_idx) + (width * height * d_idx)

##################
### SUBROUTINE ###
##################

def convert_flattened_idx_to_3d_grid_ids(idx, width, height):

    w_idx = idx % width
    h_idx = (idx // width) % height
    d_idx = idx // (width*height)

    return (w_idx, h_idx, d_idx)

##################
### SUBROUTINE ###
##################


def wrap_indices(id_arr, length, is_wrap_around=False):

    # Array to store the processed indices
    id_arr_new = []

    # Process the indices

    # If wrap around, wrap the out of domain indices into the domain
    if is_wrap_around:

        for idx in id_arr:
            was_negative = False
            if (idx < 0):
                was_negative = True
                idx = -idx

            offset = idx % length
            if was_negative:
                id_arr_new.append(length - offset)
            else:
                id_arr_new.append(offset)

    # If not wrap around just discard the out of domain indices
    else:

        for idx in id_arr:
            # If inside the domain
            if (idx >= 0) and (idx <= length-1):
                id_arr_new = np.append(id_arr_new, idx)

    # Remove repeated indices, can happen if neighborhood >= length/2
    id_arr_new = np.unique(id_arr_new)

    # Sort the array
    id_arr_new = np.sort(id_arr_new)

    return id_arr_new

##################
### SUBROUTINE ###
##################

def neigh_list_from_2d_grid_ids(
        w_idx, h_idx, width, height, is_wrap_around=False, n_w_nearest=1,
        n_h_nearest=1):

    # Sanity check: Number of nearest neighbours cannot be greater than the
    #               length of the dimension.
    # Do it in FUTURE

    # Neigh_ids
    w_neigh_ids_raw = np.linspace(w_idx - n_w_nearest, w_idx + n_w_nearest, \
        2 * n_w_nearest + 1, dtype=int)

    h_neigh_ids_raw = np.linspace(h_idx - n_h_nearest, h_idx + n_h_nearest, \
        2 * n_h_nearest + 1, dtype=int)
    
    # Process the neighbour ids to remove out of range neighbours or wrap them
    w_neigh_ids = wrap_indices(id_arr=w_neigh_ids_raw, length=width, \
        is_wrap_around=is_wrap_around)

    h_neigh_ids = wrap_indices(id_arr=h_neigh_ids_raw, length=height, \
        is_wrap_around=is_wrap_around)

    # Create a cartesian product barring self index
    neigh_list_grid_ids = []
    for neigh in itertools.product(w_neigh_ids, h_neigh_ids):
        if neigh != (w_idx, h_idx):
            neigh_list_grid_ids.append(neigh)

    return neigh_list_grid_ids

##################
### SUBROUTINE ###
##################


def neigh_list_from_2d_flattened_idx(idx, width, height, is_wrap_around=False,
                                     n_w_nearest=1, n_h_nearest=1):

    '''
            y
            |
     height |  <----------
            |             |
            |  ---------->
            |_______________________x
            0             width

    Future: Extend to nth nearest neighbours
    '''

    # Sanity check: Number of nearest neighbours cannot be greater than the length of the dimension
    # Do it in FUTURE

    # Find w_idx and h_idx
    [w_idx, h_idx] = convert_flattened_idx_to_2d_grid_ids(idx, width)

    neigh_list_grid_ids = neigh_list_from_2d_grid_ids(w_idx, h_idx, width, height, \
        is_wrap_around=is_wrap_around, n_w_nearest=n_w_nearest, \
        n_h_nearest=n_h_nearest)

    # Convert 2d grid ids to flattened ids
    neigh_list_flattened_ids = []
    for neigh in neigh_list_grid_ids:
        flattened_idx = convert_2d_grid_ids_to_flattened_idx(w_idx=neigh[0], h_idx=neigh[1], width=width)
        neigh_list_flattened_ids.append(flattened_idx)

    neigh_list_flattened_ids.sort()

    return neigh_list_flattened_ids

##################
### SUBROUTINE ###
##################


def warn_create_dir(dir_name, is_user_prompt=True):

    # Check if the directory already exists and warn
    if os.path.exists(dir_name):
        sys.stderr.write("Warning: Directory: %s already exists\n" %(dir_name))
        if is_user_prompt:
            sys.stderr.write("         Press YES to continue with overwriting" \
                " existing files\n")
            sys.stderr.write("         >>> ")
            if input() != "YES":
                sys.stderr.write("Terminating!!!\n")
                exit(1)
        else:
            sys.stderr.write("         Continuting to use the existing directory\n") 
    else:
        os.makedirs(dir_name)

##################
### SUBROUTINE ###
##################


def nonlinspace_increasing(start, stop, num, curvature=1):
    linear = np.linspace(0, 1, num)
    curve = np.sort(np.exp(-curvature*linear))
    curve = curve - np.min(curve)
    curve = curve/np.max(curve)   #  normalize between 0 and 1
    curve = curve*(stop - start) + start
    return curve

##################
### SUBROUTINE ###
##################


def nonlinspace_decreasing(start, stop, num, curvature=1):
    linear = np.linspace(0, 1, num)
    curve = 1 - np.exp(-curvature*linear)
    curve = curve/np.max(curve)   #  normalize between 0 and 1
    curve  = curve*(stop - start - 1) + start
    return curve

##################
### SUBROUTINE ###
##################


def mirrored_nonlinspace(start, stop, num, curvature=1, is_increasing=True):

    n_half = num // 2 + 1 # // is Floor division
        
    if is_increasing:
        half_list = nonlinspace_increasing(start, stop, n_half, curvature)
    else:
        half_list = nonlinspace_decreasing(start, stop, n_half, curvature)

    mirrored_list = np.concatenate((half_list, -1*half_list), axis=0)
    mirrored_list = np.sort(np.unique(mirrored_list))

    return mirrored_list

##################
### SUBROUTINE ###
##################


def asymmetric_general_spacing(left_limit, right_limit, num=None, mean=None, num_left=None, num_right=None, spacing_coefficient=1, spacing_type="increasing"):

    # Sanity check
    error_check_argument_required(
        arg_val=spacing_type, arg_name="spacing_type", module=module_name,
        subroutine="asymmetric_general_spacing", valid_args=["linear",
        "increasing", "decreasing"])

    # If mean is not given
    if mean is None:
        mean = 0.5 * (left_limit + right_limit)

    # Number of points on each wing including mean
    if num is None and (num_left is None or num_right is None):

        sys.stderr.write("Error: In module 'mod_utility.py'\n")
        sys.stderr.write("       In subroutine 'asymmetric_general_spacing'\n")
        sys.stderr.write("       Either num or both(num_left and num_right) are to be specified\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    else:
        if (num_left is None or num_right is None):
            num_left = num // 2 + 1 # // is Floor division
            num_right = num_left

    if spacing_type == "linear":
        if num_right > 0:
            right_half_list = mean + np.linspace(0.0, right_limit - mean, num_right)
        if num_left > 0:
            left_half_list = mean - np.linspace(0.0, mean - left_limit, num_left)
    elif spacing_type == "increasing":
        if num_right > 0:
            right_half_list = mean + nonlinspace_increasing(start=0.0, stop=right_limit - mean, num=num_right, curvature=spacing_coefficient)
        if num_left > 0:
            left_half_list = mean - nonlinspace_increasing(start=0.0, stop=mean - left_limit, num=num_left, curvature=spacing_coefficient)
    else:
        if num_right > 0:
            right_half_list = mean + nonlinspace_decreasing(start=0.0, stop=right_limit - mean, num=num_right, curvature=spacing_coefficient)
        if num_left > 0:
            left_half_list = mean - nonlinspace_decreasing(start=0.0, stop=mean - left_limit, num=num_left, curvature=spacing_coefficient)

    if num_right > 0 and num_left > 0:
        asymmetric_list = np.concatenate((left_half_list, right_half_list), axis=0)
    elif num_right > 0:
        asymmetric_list = right_half_list
    else:
        asymmetric_list = left_half_list

    asymmetric_list = np.sort(np.unique(asymmetric_list))

    return asymmetric_list

##################
### SUBROUTINE ###
##################


def mirrored_general_spacing(max_val, num, spacing_type, spacing_coefficient=1):

    # Sanity check
    error_check_argument_required(
        arg_val=spacing_type, arg_name="spacing_type", module=module_name,
        subroutine="mirrored_general_spacing", valid_args=["linear",
        "increasing", "decreasing"])

    if spacing_type == "linear":
        return np.linspace(-1.0*max_val, max_val, num)

    if spacing_type == "increasing":
        return mirrored_nonlinspace(
            start=0, stop=max_val, num=num, curvature=spacing_coefficient,
            is_increasing=True)

    if spacing_type == "decreasing":
        return mirrored_nonlinspace(
            start=0, stop=max_val, num=num, curvature=spacing_coefficient,
            is_increasing=False)

##################
### SUBROUTINE ###
##################


def delete_files(file_list=None):

    if file_list is None:

        file_list = basic_clean_file_list

    for f in file_list:
        try:
            os.remove(f)
        except OSError:
            pass

##################
### SUBROUTINE ###
##################


def partition_range_into_chunks_inclusive_bounds(length, n_chunks):

    chunk_size = length // n_chunks
    rem_size = length % n_chunks

    # Store results, each row has [start_idx, end_idx] inclusive bounds
    range_ids = np.zeros((n_chunks, 2), dtype=int)
    range_ids[0,0] = 0

    for i in range(0, n_chunks):

        if i < rem_size:
            range_ids[i,1] = range_ids[i,0] + chunk_size
        else:
            range_ids[i,1] = range_ids[i,0] + chunk_size - 1

        if i < n_chunks - 1:
            range_ids[i+1,0] = range_ids[i,1] + 1

    return range_ids

##################
### SUBROUTINE ###
##################

def rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4

def rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.

usage:
x,y=intersection(x1,y1,x2,y2)

    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)

    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]

##################
### SUBROUTINE ###
##################


def convert_to_rgb(minval, maxval, val):

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # [BLUE, GREEN, RED]
    # "colors" is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.
    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the "colors" pallette.
    if abs(float(maxval-minval)) < EPSILON:
        return colors[0]

    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    # Does it fall exactly on one of the color points?
    if f < EPSILON:
        return colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

##################
### SUBROUTINE ###
##################


def near(p0, point_list, d0, radius, weight_list):

    wt = 0.0
    count = 0

    for idx, p in enumerate(point_list, start=0):
        dist = mod_math.great_circle_distance_cartesian(p1=p0, p2=p, radius=radius)
        if dist < d0:
            count = count + 1
            wt = wt + weight_list[idx] * (1 - dist/d0)

    if count > 0:
        wt = wt / count

    return wt

def plot_points_weigths_on_sphere(radius, point_list, weight_list=None, option="make_data", n_theta=120, n_phi=60):

    if weight_list is None:
        weight_list = np.ones(len(point_list))

    # Make data for an opaque sphere or spherical grid
    u = np.linspace(0, 2 * np.pi, n_theta)
    v = np.linspace(0, np.pi, n_phi)
    sx = 1.0 * radius * np.outer(np.cos(u), np.sin(v))
    sy = 1.0 * radius * np.outer(np.sin(u), np.sin(v))
    sz = 1.0 * radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ds = 1.0 * (np.pi * radius / n_phi)

    if option == "grid_contour":

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        fig = plt.figure(figsize=(8.0, 5.0))
        ax = plt.axes(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        WW = sx.copy()
        for i in range( len( sx ) ):
            for j in range( len( sx[0] ) ):
                x = sx[i,j]
                y = sy[i,j]
                z = sz[i,j]
                WW[i,j] = near(np.array([x, y, z ]), point_list, ds, radius, weight_list)
        print(np.amin(WW))
        print(np.amax(WW))
        WW = WW / np.amax(WW)
        myheatmap = WW

        ax.plot_surface(sx, sy, sz, cstride=1, rstride=1, facecolors=cm.jet(myheatmap))

        ax.set_xlim([-radius,radius])
        ax.set_ylim([-radius,radius])
        ax.set_zlim([-radius,radius])
        plt.tight_layout()
        plt.show()

    elif option == "point_sphere":

        from mayavi import mlab

        # Create a sphere
        r = 1.0
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

        x = r*sin(phi)*cos(theta)
        y = r*sin(phi)*sin(theta)
        z = r*cos(phi)

        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
        mlab.clf()

        xx, yy, zz = np.hsplit(point_list, 3) 
        mlab.mesh(x , y , z, color=(0.0,0.5,0.5))
        mlab.points3d(xx, yy, zz, scale_factor=0.05)
        mlab.show()

    elif option == "make_data":

        # Create rgb from weights
        rgb_list = np.zeros((len(point_list),3), dtype=int)
        min_wt = np.amin(weight_list)
        max_wt = np.amax(weight_list)
        for idx, wt in enumerate(weight_list, start=0):
            rgb_list[idx] = convert_to_rgb(minval=min_wt, maxval=max_wt, val=wt)

        return 

##################
### SUBROUTINE ###
##################


def cluster_means(arr, tol):

    s_arr = np.sort(arr)
    return [group.mean() for group in np.split(s_arr, np.where(np.diff(s_arr) > tol)[0]+1)]

##################
### SUBROUTINE ###
##################


def get_real_string(rnum, n_decimals=2, decimal_separator="_"):

    if rnum < 0.0:
        return 'NEG_' + str(round(abs(rnum), n_decimals)).replace(".",decimal_separator)
    else:
        return 'POS_' + str(round(abs(rnum), n_decimals)).replace(".",decimal_separator)

##################
### SUBROUTINE ###
##################


def fit_bspline(x, y, N=100):
    t, c, k = interpolate.splrep(x, y, s=0, k=1)
    xmin, xmax = x.min(), x.max()
    xx = np.linspace(xmin, xmax, N)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    return (xx, spline)

##################
### SUBROUTINE ###
##################


def get_smoothed_intersection(x1_arr, y1_arr, x2_arr, y2_arr, order=1, n_points=300, is_clean=True):

    if is_clean:
        # Only consider data that is monotonically increasing
        idx_1 = 0
        for idx in range(1, len(x1_arr)):
            if x1_arr[idx-1] > x1_arr[idx]:
                break
            if y1_arr[idx-1] > y1_arr[idx]:
                break
            idx_1 = idx_1 + 1

        idx_2 = 0
        for idx in range(1, len(x2_arr)):
            if x2_arr[idx-1] > x2_arr[idx]:
                break
            if y2_arr[idx-1] > y2_arr[idx]:
                break
            idx_2 = idx_2 + 1
    else:
        idx_1 = len(x1_arr)
        idx_2 = len(x2_arr)

    # Smooth the data (s for smooth) of deviatoric radius
    [x1_s_arr, y1_s_arr] = smooth_plot(x_arr=x1_arr[0:idx_1+1], y_arr=y1_arr[0:idx_1+1], order=order, n_points=n_points)
    [x2_s_arr, y2_s_arr] = smooth_plot(x_arr=x2_arr[0:idx_2+1], y_arr=y2_arr[0:idx_2+1], order=order, n_points=n_points)

    # Finding the intersection point between parent and daughter curves
    # NOTE: Multiple intersections can exist
    [x_int_arr, y_int_arr] = intersection(x1=x1_s_arr, y1=y1_s_arr, x2=x2_s_arr, y2=y2_s_arr)

    if len(x_int_arr) == 0:
        x_int_arr = [None]
        y_int_arr = [None]

    return (x_int_arr, y_int_arr)

##################
### SUBROUTINE ###
##################

def get_nearest_indices(array, value, tol):

    '''
    Returns the list of indices whose array values lie within tolerance to the
    specified cvalue
    '''
    nearest_ids = []
    for idx, array_value in enumerate(array, start=0):
        if math.isclose(value, array_value, abs_tol=tol):
            nearest_ids.append(idx)

    return nearest_ids

##################
### SUBROUTINE ###
##################

def round_to_array_extrema(arr, val, abs_tol=0.01):

    min_val = np.amin(arr)
    max_val = np.amax(arr)

    if (val < min_val) and math.isclose(min_val, val, abs_tol=abs_tol):
        return min_val
    elif (val > max_val) and math.isclose(max_val, val, abs_tol=abs_tol):
        return max_val
    else:
        return val

##################
### SUBROUTINE ###
##################


def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(0, size_x):
        matrix [x, 0] = x
    for y in range(0, size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

##################
### SUBROUTINE ###
##################


def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        #return "The strings are {} edits away".format(distance[row][col])
        return distance[row][col]

##################
### SUBROUTINE ###
##################


def get_bin_open_close_idx(bin_min, bin_max, n_bins, value):

    '''
    Bins value according to: [      ](       ](       ] ...... (        ]
    '''

    if value < bin_min:
        sys.stderr.write("Error: In module 'mod_utility.py'\n")
        sys.stderr.write("       In subroutine 'get_bin_open_close_idx'\n")
        sys.stderr.write("       'value' less than bin lower bound\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    if value > bin_max:
        sys.stderr.write("Error: In module 'mod_utility.py'\n")
        sys.stderr.write("       In subroutine 'get_bin_open_close_idx'\n")
        sys.stderr.write("       'value' greater than bin upper bound\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    frac = value / (bin_max - bin_min)
    bidx = math.floor(n_bins * frac)
    uidx = math.ceil(n_bins * frac)
    if (uidx != 0):
        if (bidx == uidx):
            bidx = bidx - 1

    return bidx

##################
### SUBROUTINE ###
##################


def hex_to_RGB255(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]
  
##################
### SUBROUTINE ###
##################


def hex_to_RGBfloat(hex):
  ''' "#FFFFFF" -> [1.0, 1.0, 1.0] '''
  # Pass 16 to the integer function for change of base
  return RGB255_to_RGBfloat(hex_to_RGB255(hex))

##################
### SUBROUTINE ###
##################


def RGB255_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])
            
##################
### SUBROUTINE ###
##################

def RGB255_to_RGBfloat(RGB):

    return [x / 255.0 for x in RGB]


##################
### SUBROUTINE ###
##################


# Draw polygon with linear gradient from point 1 to point 2 and ranging
# from color 1 to color 2 on given image
def linear_gradient(i, poly, p1, p2, c1, c2):

    # Draw initial polygon, alpha channel only, on an empty canvas of image size
    ii = Image.new('RGBA', i.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ii)
    draw.polygon(poly, fill=(0, 0, 0, 255), outline=None)

    # Calculate angle between point 1 and 2
    p1 = np.array(p1)
    p2 = np.array(p2)
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) / np.pi * 180

    # Rotate and crop shape
    temp = ii.rotate(angle, expand=True)
    temp = temp.crop(temp.getbbox())
    wt, ht = temp.size

    # Create gradient from color 1 to 2 of appropriate size
    gradient = np.linspace(c1, c2, wt, True).astype(np.uint8)
    gradient = np.tile(gradient, [2 * h, 1, 1])
    gradient = Image.fromarray(gradient)

    # Paste gradient on blank canvas of sufficient size
    temp = Image.new('RGBA', (max(i.size[0], gradient.size[0]),
                              max(i.size[1], gradient.size[1])), (0, 0, 0, 0))
    temp.paste(gradient)
    gradient = temp

    # Rotate and translate gradient appropriately
    x = np.sin(angle * np.pi / 180) * ht
    y = np.cos(angle * np.pi / 180) * ht
    gradient = gradient.rotate(-angle, center=(0, 0),
                               translate=(p1[0] + x, p1[1] - y))

    # Paste gradient on temporary image
    ii.paste(gradient.crop((0, 0, ii.size[0], ii.size[1])), mask=ii)

    # Paste temporary image on actual image
    i.paste(ii, mask=ii)

    return i


##################
### SUBROUTINE ###
##################


# Draw polygon with radial gradient from point to the polygon border
# ranging from color 1 to color 2 on given image
def radial_gradient(i, poly, p, c1, c2):

    # Draw initial polygon, alpha channel only, on an empty canvas of image size
    ii = Image.new('RGBA', i.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ii)
    draw.polygon(poly, fill=(0, 0, 0, 255), outline=None)

    # Use polygon vertex with highest distance to given point as end of gradient
    p = np.array(p)
    max_dist = max([np.linalg.norm(np.array(v) - p) for v in poly])

    # Calculate color values (gradient) for the whole canvas
    x, y = np.meshgrid(np.arange(i.size[0]), np.arange(i.size[1]))
    c = np.linalg.norm(np.stack((x, y), axis=2) - p, axis=2) / max_dist
    c = np.tile(np.expand_dims(c, axis=2), [1, 1, 3])
    c = (c1 * (1 - c) + c2 * c).astype(np.uint8)
    c = Image.fromarray(c)

    # Paste gradient on temporary image
    ii.paste(c, mask=ii)

    # Paste temporary image on actual image
    i.paste(ii, mask=ii)

    return i
    
##################
### SUBROUTINE ###
##################

def vegard_latparam(p1, p2, x):

    '''
    Linear interpolation of lattice parameter
    '''

    # Sanity check 1
    if (p1 is None) and (p2 is None):

        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'vegard_latparam'\n")
        sys.stderr.write("       Both lattice paremeters 'p1' and 'p2' cannot be None.\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
    
    # Sanity check 2
    if (x < 0.0) or (x > 1.0):

        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'vegard_latparam'\n")
        sys.stderr.write("       The mixing parameter should be in the range of [0, 1.0].\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
        
    if p1 is None:

        return p2

    elif p2 is None:

        return p1

    else:

        return x * p1 + (1.0-x) * p2



'''----------------------------------------------------------------------------
                              END OF MODULE
----------------------------------------------------------------------------'''
