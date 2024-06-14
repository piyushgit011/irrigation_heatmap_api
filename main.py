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
from scipy.interpolate import Rbf, interp2d
from scipy.ndimage import gaussian_filter

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get('/')
def root():
    return {"message": "Working"}

i = 0
@app.post('/heat_map')
async def heat_map(ir: IRSchema):
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', ['#E6FFE6', 'green'], 5)
    
    data = np.asarray(ir.array)
    x, y = np.arange(data.shape[1]), np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)
    points = np.array([X.flatten(), Y.flatten()]).T
    values = data.flatten()
    xi, yi = np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500)
    XI, YI = np.meshgrid(xi, yi)
    interp_spline = interp2d(x, y, data, kind='linear')  # Can change kind to 'linear', 'cubic', etc.
    zi = interp_spline(xi, yi)
    zi = gaussian_filter(zi, sigma=3)

    # Custom colormap and normalization
    vmax = values.max()
    non_zero_values = values[values!= 0]
    if non_zero_values.size > 0:
        vmin = non_zero_values.min() - 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = ["yellow","limegreen", "green", "darkgreen"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    cmap.set_under('white')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(XI, YI, zi, levels=np.linspace(vmin, vmax, 256), cmap=cmap, norm=norm, extend='both')
    contours = ax.contour(XI, YI, zi, levels=np.linspace(vmin, vmax, 60), colors='none')

    for contour_path in contours.collections[0].get_paths():
        vertices = contour_path.vertices
        tck, u = splprep([vertices[:, 0], vertices[:, 1]], s=5)
        new_points = splev(np.linspace(0, 1, 1000), tck)
        smooth_contour = Path(list(zip(new_points[0], new_points[1])), [Path.MOVETO] + [Path.CURVE4] * (len(new_points[0]) - 1))
        patch = PathPatch(smooth_contour, facecolor='none', edgecolor='grey', linewidth=2)
        ax.add_patch(patch)

    # Finding nonzero data bounds for adjusting the plot limits
    nonzero_indices = np.nonzero(data)
    min_x, max_x = x[nonzero_indices[1]].min(), x[nonzero_indices[1]].max()
    min_y, max_y = y[nonzero_indices[0]].min(), y[nonzero_indices[0]].max()
    padding = 2
    min_x, max_x = max(min_x - padding, 0), min(max_x + padding, data.shape[1]-1)
    min_y, max_y = max(min_y - padding, 0), min(max_y + padding, data.shape[0]-1)

    # Adjust plot limits
    plt.xlim(xi[max(int(min_x * 499 / (x.max()+1)), 0)], xi[min(int(max_x * 499 / (x.max()+1)), 499)])
    plt.ylim(yi[max(int(min_y * 499 / (y.max()+1)), 0)], yi[min(int(max_y * 499 / (y.max()+1)), 499)])

    ax.set_aspect('auto')
    ax.axis('off')
    ax.invert_yaxis()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)  # Rewind the buffer

    timestamp = int(time.time())
    filename = f'heatmaps/green_only_heatmap{timestamp}.png'
    url = upload_image_to_supabase(buf, filename)  # Read bytes from buffer

    plt.close(fig)
    buf.close()
    # fig.savefig('static/green_only_heatmap.png', bbox_inches='tight', pad_inches=0)
    

    # url = convert_to_url_file('static/green_only_heatmap.png', f'maps/green_only_heatmap{int(time.time())}.png')

    non_zero_indices = np.nonzero(data)
    non_zero_values = data[non_zero_indices]
    sorted = np.sort(non_zero_values)
    
    lowest_quartile = round(sorted.size/4)
    lowest_quartile_avg = np.average(sorted[:lowest_quartile])
    overall_average = np.average(sorted)
    du = (lowest_quartile_avg/overall_average)*100
    
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

    uvicorn.run(app, host="0.0.0.0", port=4000)
