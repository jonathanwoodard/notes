import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PatchCollection
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
import fiona
from descartes import PolygonPatch

################################################################################

df_map = pd.DataFrame({
    'poly': [Polygon(hood_points) for hood_points in m.seattle],
    'name': [hood['S_HOOD'] for hood in m.seattle_info]
})
# Convert our latitude and longitude into Basemap cartesian map coordinates
mapped_points = [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(ld['longitude'],
            ld['latitude'])]
all_points = MultiPoint(mapped_points)
# Use prep to optimize polygons for faster computation
hood_polygons = prep(MultiPolygon(list(df_map['poly'].values)))
# Filter out the points that do not fall within the map we're making
city_points = filter(hood_polygons.contains, all_points)
def num_of_contained_points(apolygon, city_points):
    return int(len(filter(prep(apolygon).contains, city_points)))

df_map['hood_count'] = df_map['poly'].apply(num_of_contained_points, args=(city_points,))
df_map['hood_hours'] = df_map.hood_count / 60.0


# Check out the full post at http://beneathdata.com/how-to/visualizing-my-location-history/
# to utilize the code below

# We'll only use a handful of distinct colors for our choropleth. So pick where
# you want your cutoffs to occur. Leave zero and ~infinity alone.
breaks = [0.] + [4., 24., 64., 135.] + [1e20]


def self_categorize(entry, breaks):
    for i in range(len(breaks) - 1):
        if entry > breaks[i] and entry <= breaks[i + 1]:
            return i
    return -1
df_map['jenks_bins'] = df_map.hood_hours.apply(self_categorize, args=(breaks,))

labels = ['Never been\nhere'] + ["> %d hours" % (perc) for perc in breaks[:-1]]

# Or, you could always use Natural_Breaks to calculate your breaks for you:
# from pysal.esda.mapclassify import Natural_Breaks
# breaks = Natural_Breaks(df_map[df_map['hood_hours'] > 0].hood_hours, initial=300, k=3)
# df_map['jenks_bins'] = -1 #default value if no data exists for this bin
# df_map['jenks_bins'][df_map.hood_count > 0] = breaks.yb
#
# jenks_labels = ['Never been here', "> 0 hours"]+["> %d hours"%(perc) for perc in breaks.bins[:-1]]


def custom_colorbar(cmap, ncolors, labels, **kwargs):
    """Create a custom, discretized colorbar with correctly formatted/aligned labels.

    cmap: the matplotlib colormap object you plan on using for your graph
    ncolors: (int) the number of discrete colors available
    labels: the list of labels for the colorbar. Should be the same length as ncolors.
    """
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable

    norm = BoundaryNorm(range(0, ncolors), cmap.N)
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors + 1) + 0.5)
    colorbar.set_ticklabels(range(0, ncolors))
    colorbar.set_ticklabels(labels)
    return colorbar

figwidth = 14
fig = plt.figure(figsize=(figwidth, figwidth * h / w))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

cmap = plt.get_cmap('Blues')
# draw neighborhoods with grey outlines
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
pc = PatchCollection(df_map['patches'], match_original=True)
# apply our custom color values onto the patch collection
cmap_list = [cmap(val) for val in (df_map.jenks_bins.values - df_map.jenks_bins.values.min()) / (
             df_map.jenks_bins.values.max() - float(df_map.jenks_bins.values.min()))]
pc.set_facecolor(cmap_list)
ax.add_collection(pc)

# Draw a map scale
m.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
               coords[0], coords[1], 10.,
               fontsize=16, barstyle='fancy', labelstyle='simple',
               fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
               zorder=5, ax=ax,)

# ncolors+1 because we're using a "zero-th" color
cbar = custom_colorbar(cmap, ncolors=len(jenks_labels) + 1, labels=jenks_labels, shrink=0.5)
cbar.ax.tick_params(labelsize=16)

fig.suptitle("Time Spent in Seattle Neighborhoods", fontdict={'size': 24, 'fontweight': 'bold'}, y=0.92)
ax.set_title("Using location data collected from my Android phone via Google Takeout", fontsize=14, y=0.98)
qax.text(1.35, 0.04, "Collected from 2012-2014 on Android 4.2-4.4\nGeographic data provided by data.seattle.gov",
         ha='right', color='#555555', style='italic', transform=ax.transAxes)
ax.text(1.35, 0.01, "BeneathData.com", color='#555555', fontsize=16, ha='right', transform=ax.transAxes)

plt.savefig('chloropleth.png', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5, facecolor='#F2F2F2')

################################################################################

"""PLOT A HEXBIN MAP OF LOCATION
"""
figwidth = 14
fig = plt.figure(figsize=(figwidth, figwidth*h/w))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

# draw neighborhood patches from polygons
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
    x, fc='#555555', ec='#555555', lw=1, alpha=1, zorder=0))
# plot neighborhoods by adding the PatchCollection to the axes instance
ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))

# the mincnt argument only shows cells with a value >= 1
# The number of hexbins you want in the x-direction
numhexbins = 50
hx = m.hexbin(
    np.array([geom.x for geom in city_points]),
    np.array([geom.y for geom in city_points]),
    gridsize=(numhexbins, int(numhexbins*h/w)), #critical to get regular hexagon, must stretch to map dimensions
    bins='log', mincnt=1, edgecolor='none', alpha=1.,
    cmap=plt.get_cmap('Blues'))

# Draw the patches again, but this time just their borders (to achieve borders over the hexbins)
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
    x, fc='none', ec='#FFFF99', lw=1, alpha=1, zorder=1))
ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))

# Draw a map scale
m.drawmapscale(coords[0] + 0.05, coords[1] - 0.01,
    coords[0], coords[1], 4.,
    units='mi', barstyle='fancy', labelstyle='simple',
    fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
    zorder=5)

fig.suptitle("My location density in Seattle", fontdict={'size': 24, 'fontweight': 'bold'}, y=0.92)
ax.set_title("Using location data collected from my Android phone via Google Takeout", fontsize=14, y=0.98)
ax.text(1.0, 0.03, "Collected from 2012-2014 on Android 4.2-4.4\nGeographic data provided by data.seattle.gov",
        ha='right', color='#555555', style='italic', transform=ax.transAxes)
ax.text(1.0, 0.01, "BeneathData.com", color='#555555', fontsize=16, ha='right', transform=ax.transAxes)
plt.savefig('hexbin.png', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5, facecolor='#DEDEDE')


################################################################################

"""
This short code snippet utilizes the new animation package in
matplotlib 1.1.0; it's the shortest snippet that I know of that can
produce an animated plot in python. I'm still hoping that the
animate package's syntax can be simplified further.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def simData():
    # this function is called as the argument for
    # the simPoints function. This function contains
    # (or defines) and iterator---a device that computes
    # a value, passes it back to the main program, and then
    # returns to exactly where it left off in the function upon the
    # next call. I believe that one has to use this method to animate
    # a function using the matplotlib animation package.
    #
    t_max = 10.0
    dt = 0.05
    x = 0.0
    t = 0.0
    while t < t_max:
        x = np.sin(np.pi * t)
        t = t + dt
        yield x, t


def simPoints(simData):
    x, t = simData[0], simData[1]
    time_text.set_text(time_template % (t))
    line.set_data(t, x)
    return line, time_text

#
#   set up figure for plotting:
#
fig = plt.figure()
ax = fig.add_subplot(111)
# I'm still unfamiliar with the following line of code:
line, = ax.plot([], [], 'bo', ms=10)
ax.set_ylim(-1, 1)
ax.set_xlim(0, 10)
#
time_template = 'Time = %.1f s'    # prints running simulation time
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# Now call the animation package: (simData is the user function
# serving as the argument for simPoints):
ani = animation.FuncAnimation(fig, simPoints, simData, blit=False,
                              interval=10, repeat=True)
plt.show()

################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

# This example uses subclassing, but there is no reason that the proper function
# couldn't be set up and then use FuncAnimation. The code is long, but not
# really complex. The length is due solely to the fact that there are a total
# of 9 lines that need to be changed for the animation as well as 3 subplots
# that need initial set up.

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)

        self.t = np.linspace(0, 80, 400)
        self.x = np.cos(2 * np.pi * self.t / 10.)
        self.y = np.sin(2 * np.pi * self.t / 10.)
        self.z = 10 * self.t

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        self.line1 = Line2D([], [], color='black')
        self.line1a = Line2D([], [], color='red', linewidth=2)
        self.line1e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        ax1.add_line(self.line1)
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal', 'datalim')

        ax2.set_xlabel('y')
        ax2.set_ylabel('z')
        self.line2 = Line2D([], [], color='black')
        self.line2a = Line2D([], [], color='red', linewidth=2)
        self.line2e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        ax2.add_line(self.line2)
        ax2.add_line(self.line2a)
        ax2.add_line(self.line2e)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(0, 800)

        ax3.set_xlabel('x')
        ax3.set_ylabel('z')
        self.line3 = Line2D([], [], color='black')
        self.line3a = Line2D([], [], color='red', linewidth=2)
        self.line3e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        ax3.add_line(self.line3)
        ax3.add_line(self.line3a)
        ax3.add_line(self.line3e)
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(0, 800)

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1
        head_len = 10
        head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

        self.line1.set_data(self.x[:i], self.y[:i])
        self.line1a.set_data(self.x[head_slice], self.y[head_slice])
        self.line1e.set_data(self.x[head], self.y[head])

        self.line2.set_data(self.y[:i], self.z[:i])
        self.line2a.set_data(self.y[head_slice], self.z[head_slice])
        self.line2e.set_data(self.y[head], self.z[head])

        self.line3.set_data(self.x[:i], self.z[:i])
        self.line3a.set_data(self.x[head_slice], self.z[head_slice])
        self.line3e.set_data(self.x[head], self.z[head])

        self._drawn_artists = [self.line1, self.line1a, self.line1e,
            self.line2, self.line2a, self.line2e,
            self.line3, self.line3a, self.line3e]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines =  [self.line1, self.line1a, self.line1e,
            self.line2, self.line2a, self.line2e,
            self.line3, self.line3a, self.line3e]
        for l in lines:
            l.set_data([], [])

ani = SubplotAnimation()
#ani.save('test_sub.mp4')
plt.show()
