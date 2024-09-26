'''-----------------------------------------------------------------------------
                                  mod_colors.py

 Description: Colour related utility.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
import matplotlib.pyplot as plt
from collections import namedtuple

# Externally installed modules
# None

# Local imports
# None

'''-----------------------------------------------------------------------------
                                MODULE VARIABLES
-----------------------------------------------------------------------------'''

# Bright colors
line_colors = namedtuple('line_colors', 'blue red green yellow cyan purple magenta teal orange olive grey black')
line_color_set = line_colors('#0077BB', '#CC3311', '#228833', '#DDAA33', '#66CCEE', '#AA3377', '#FF00FF', '#44AA99', '#EE7733', '#999933', '#BBBBBB', '#000000')

'''-----------------------------------------------------------------------------
                                TEST MODULE
-----------------------------------------------------------------------------'''
if __name__ == '__main__':

    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(frame_on=True)
    
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top=False, labelsize=12)
    ax.xaxis.set_tick_params(which='minor', size=2.5, width=0.5, direction='in', top=False)
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right=False, labelsize=12)
    ax.yaxis.set_tick_params(which='minor', size=2.5, width=0.5, direction='in', right=False)

    h = 0.1
    for (idx, line_color) in enumerate(line_color_set):
        plt.plot( [0, 10], [idx*h, idx*h], color=line_color, linewidth=3.5)
    
    plt.savefig('colors_test.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()

'''-----------------------------------------------------------------------------
                                END OF MODULE
-----------------------------------------------------------------------------'''
