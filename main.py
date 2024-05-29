import uvicorn
from fastapi import FastAPI
from irrigationSchema import Irrigation as IRSchema
import base64
import numpy as np
import json
import matplotlib as mpl
import seaborn as sns
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from matplotlib.colors import ListedColormap, BoundaryNorm
# from firebase import upload_to_cloud_from_memory
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path
from scipy.interpolate import splprep, splev
from matplotlib.patches import PathPatch
import io 
from supa import upload_image_to_supabase
app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get('/')
def root():
    return {"message": "Working"}

i = 0
@app.post('/heat_map')
async def heat_map(ir: IRSchema):
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', ['#E6FFE6', 'green'], 5)
    
    arr = np.asarray(ir.array)
    data = arr
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)

# Flatten and interpolate as before
    points = np.array([X.flatten(), Y.flatten()]).T
    values = data.flatten()
    xi = np.linspace(x.min(), x.max(), 500)
    yi = np.linspace(y.min(), y.max(), 500)
    XI, YI = np.meshgrid(xi, yi)
    zi = griddata(points, values, (XI, YI), method='cubic')

# Create custom colormap and normalization
    vmin, vmax = 7, 30
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = ["yellow", "lime" ,"limegreen", "green","darkgreen"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# Define main plot axes to use full width since there's no colorbar
    fig = plt.figure(figsize=(10, 8))
    main_axes = fig.add_axes([0.1, 0.1, 0.85, 0.8])  # Adjusted to use more width
    contourf = main_axes.contourf(XI, YI, zi, levels=np.linspace(vmin, vmax, 256), cmap=cmap, norm=norm)

# Calculate aspect based on the data ranges
    aspect_ratio = (yi.max() - yi.min()) / (xi.max() - xi.min())
    main_axes.set_aspect(aspect_ratio)  # Adjust aspect ratio based on your data range
    contours = plt.contour(XI, YI, zi, levels=np.linspace(vmin, vmax, 60), colors='none')
# Remaining plotting and smoothing operations
# Add your contour and smoothing code here
    for contour_path in contours.collections[0].get_paths():
      vertices = contour_path.vertices
      tck, u = splprep([vertices[:, 0], vertices[:, 1]], s=5)  # Increased smoothing factor
      new_points = splev(np.linspace(0, 1, 1000), tck)  # Increased number of evaluation points
      smooth_contour = Path(list(zip(new_points[0], new_points[1])), [Path.MOVETO] + [Path.CURVE4] * (len(new_points[0]) - 1))
      patch = PathPatch(smooth_contour, facecolor='none', edgecolor='grey', linewidth=2)
      plt.gca().add_patch(patch)
    main_axes.axis('off')
    main_axes.invert_yaxis()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)  # Rewind the buffer

    timestamp = int(time.time())
    filename = f'green_only_heatmap{timestamp}.png'
    url = upload_image_to_supabase(buf, filename)  # Read bytes from buffer

    plt.close(fig)
    buf.close()
    # fig.savefig('static/green_only_heatmap.png', bbox_inches='tight', pad_inches=0)
    

    # url = convert_to_url_file('static/green_only_heatmap.png', f'maps/green_only_heatmap{int(time.time())}.png')

    non_zero_indices = np.nonzero(arr)
    non_zero_values = arr[non_zero_indices]
    sorted = np.sort(non_zero_values)
    
    lowest_quartile = round(sorted.size/4)
    lowest_quartile_avg = np.average(sorted[:lowest_quartile])
    overall_average = np.average(sorted)
    du = lowest_quartile_avg/overall_average
    
    # ax = sns.heatmap(arr,cmap=cmap,  cbar=False)
    # fig = ax.get_figure()
    # fig.savefig('./static/sns.png')
    
    # fig=plt.figure()
    # ax=fig.add_subplot(1,1,1)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # plt.axis('off')
    # plt.imshow(arr, cmap='jet', interpolation='lanczos')
    # plt.savefig('./static/interpolated_heatmap.png', bbox_inches='tight',transparent=True, pad_inches=0)
    
    
    return {"message": "done",
            "number_of_cans": non_zero_values.size,
            "lowest_quartile": lowest_quartile,
            "lowest_quartile_avg":lowest_quartile_avg,
            "overall_average":overall_average, "du": du,
            "interpolated_map": f"{url}"
            }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",reload=True, port=4000)
