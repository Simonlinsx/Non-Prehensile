"""Recorded RGB beside actual 3-D semantic geometry; no invented volumetric planner.

Uses frozen collision meshes and snapshot poses. The existing 2-D placement cost
remains a distinct layer. HTML is self-contained and works without a CDN.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from scipy.spatial.transform import Rotation
import trimesh

from fr3_semantic_cost_map import polygon
from render_fr3_semantic_map import project


def main(folder):
    folder = folder.resolve()
    d = json.loads((folder / 'map_snapshot.json').read_text())
    launch = json.loads((folder / 'launch.json').read_text())
    manifest = json.loads((folder / 'manifest.jsonl').read_text())
    assets = Path(launch['environment']['FR3_SEMANTIC_ASSETS'])
    rgb = np.asarray(Image.open(folder / 'scene_original.png'))
    h, w = rgb.shape[:2]
    camera = d['camera']
    points = np.asarray(d['target_points'])
    semantic = np.asarray(d['target_semantics'])
    protected = semantic[:, 1] >= .25
    safe = (semantic[:, 0] >= .25) & ~protected
    colors = np.full(len(points), '#7c8797', dtype='<U7')
    colors[safe] = '#15965b'
    colors[protected] = '#cf2345'
    meshes = []
    hashes = {}
    for rule in d['obstacles']:
        path = assets / rule['asset_id'].split(':')[0] / 'geometry.obj'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == launch['input_hashes'][str(path)], 'Frozen mesh changed'
        hashes[str(path)] = digest
        mesh = trimesh.load(path, force='mesh', process=False)
        assert isinstance(mesh, trimesh.Trimesh)
        spec = next(o for o in manifest['objects'] if o['asset_id'] == rule['asset_id'])
        q = np.asarray(rule['quaternion_wxyz'])
        vertices = Rotation.from_quat(q[[1, 2, 3, 0]]).apply(
            np.asarray(mesh.vertices) * spec['scale']) + rule['position']
        # Three-dimensional geometry must reproduce the runtime XY footprint.
        assert np.allclose(polygon(vertices), rule['polygon'], atol=1e-8)
        meshes.append((rule, vertices, np.asarray(mesh.faces)))

    pose = np.asarray(d['target_pose'])
    goal = np.asarray(d['goal'])
    goal_points = Rotation.from_quat(goal[[4, 5, 6, 3]]).apply(
        Rotation.from_quat(pose[[4, 5, 6, 3]]).inv().apply(points - pose[:3])) + goal[:3]
    all_points = np.concatenate([points, goal_points] + [v for _, v, _ in meshes])
    lo, hi = all_points.min(0), all_points.max(0)
    uv = project(all_points, camera, w, h)
    crop = [max(0, uv[:, 0].min() - 65), min(w, uv[:, 0].max() + 65),
            max(0, uv[:, 1].min() - 55), min(h, uv[:, 1].max() + 45)]

    # Static export uses exactly the recorded USD camera, not an approximate
    # matplotlib 3-D viewing angle. Triangle depth/shading exposes real height.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#fafbfc')
    fig.subplots_adjust(left=.02, right=.98, top=.80, bottom=.18, wspace=.04)
    axes[0].imshow(rgb)
    axes[1].imshow(rgb, alpha=.12)
    triangles, shades, depths = [], [], []
    for i, (rule, vertices, faces) in enumerate(meshes, 1):
        color = '#f0b832' if rule['mode'] == 'acceptable' else '#db3b52'
        xyz = vertices[faces]
        normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        light = np.array([.3, -.4, .866])
        brightness = .50 + .50 * np.abs(normal @ light)
        triangles.extend(project(vertices, camera, w, h)[faces])
        shades.extend(np.asarray(to_rgb(color))[None, :] * brightness[:, None])
        depths.extend((np.c_[xyz.mean(1), np.ones(len(faces))] @ camera['view'])[:, 2])
        label = project([vertices.mean(0)], camera, w, h)[0]
        for ax in axes:
            ax.annotate(f'{i}' if ax is axes[0] else f'{i}  {rule["id"]}', label,
                        xytext=(0, 25), textcoords='offset points', ha='center',
                        fontsize=10, weight='bold', zorder=8,
                        bbox=dict(fc='white', ec='none', alpha=.92, pad=3))
    order = np.argsort(depths)
    axes[1].add_collection(PolyCollection(np.asarray(triangles)[order],
        facecolors=np.asarray(shades)[order], edgecolors='none', zorder=3))
    pu = project(points, camera, w, h)
    axes[1].scatter(pu[:, 0], pu[:, 1], c=colors, s=5, linewidths=0, zorder=4)
    gu = project(goal_points, camera, w, h)
    axes[1].scatter(gu[:, 0], gu[:, 1], c='#05a6c8', s=2, alpha=.20, linewidths=0, zorder=2)
    for ax, title in zip(axes, ['Original IsaacLab scene', '3-D semantic geometry | same camera']):
        ax.set_xlim(crop[:2]); ax.set_ylim(crop[3], crop[2]); ax.axis('off')
        ax.set_title(title, loc='left', fontsize=16, weight='bold', pad=12)
    fig.suptitle('Contact semantics on actual 3-D surfaces', fontsize=19, weight='bold')
    fig.legend(handles=[Patch(color='#15965b', label='Tool: legal robot contact'),
        Patch(color='#cf2345', label='Tool: protected part'),
        Patch(color='#f0b832', label='Doll: acceptable contact'),
        Patch(color='#db3b52', label='Cup / balloon: forbidden')],
        loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5,.04))
    fig.text(.5, .015, 'Cyan = goal pose. Surface relations shown here; the planar placement-cost slice is a separate layer.',
             ha='center', fontsize=9, color='#354354')
    fig.savefig(folder / 'scene_vs_semantic_3d.png', dpi=180, bbox_inches='tight')
    fig.savefig(folder / 'scene_vs_semantic_3d.pdf', bbox_inches='tight')
    plt.close(fig)

    plot = go.Figure()
    for rule, vertices, faces in meshes:
        color = '#f0b832' if rule['mode'] == 'acceptable' else '#db3b52'
        plot.add_trace(go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2], color=color, opacity=.90,
            name=f'{rule["id"]}: {rule["mode"]}', showlegend=True,
            hovertemplate=f'{rule["id"]}<br>Robot / tool contact: {rule["mode"]}<extra></extra>'))
    for mask, color, name in [(safe,'#15965b','Hammer: legal contact'),
        (protected,'#cf2345','Hammer: protected part'),(~(safe|protected),'#7c8797','Hammer: unlabelled')]:
        p = points[mask]
        plot.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',
            marker=dict(size=3,color=color),name=name))
    plot.add_trace(go.Scatter3d(x=goal_points[:,0],y=goal_points[:,1],z=goal_points[:,2],
        mode='markers',marker=dict(size=2,color='#05a6c8',opacity=.22),name='Goal pose'))
    # The very same consumed planar placement costs are optional at z=0.
    slices = np.load(folder/'cost_map_slices.npz')
    x, y = np.meshgrid(slices['x'], slices['y'])
    plot.add_trace(go.Surface(x=x, y=y, z=np.zeros_like(x), surfacecolor=slices['classes'][0],
        cmin=0,cmax=2,colorscale=[[0,'#80cc9e'],[.249,'#80cc9e'],[.25,'#f3c64d'],
        [.749,'#f3c64d'],[.75,'#ec7779'],[1,'#ec7779']], showscale=False,
        opacity=.42,name='2-D placement cost at current yaw',showlegend=True,visible='legendonly',
        hovertemplate='Planar placement slice, z=0<br>Class: %{surfacecolor}<extra></extra>'))
    inv_view = np.linalg.inv(np.asarray(camera['view']))
    eye = inv_view[3,:3] - (lo+hi)/2
    eye = eye / np.linalg.norm(eye) * 1.8
    plot.update_layout(margin=dict(l=0,r=0,t=5,b=0),paper_bgcolor='#fafbfc',
        legend=dict(orientation='h',y=-.03,font=dict(size=11)),
        scene=dict(aspectmode='data',camera=dict(eye=dict(zip('xyz',eye)),up=dict(x=0,y=0,z=1)),
        xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)',
        xaxis=dict(range=[lo[0]-.04,hi[0]+.04]),yaxis=dict(range=[lo[1]-.04,hi[1]+.04]),
        zaxis=dict(range=[-.005,hi[2]+.02])))
    img64 = base64.b64encode((folder/'scene_original.png').read_bytes()).decode()
    graph = plot.to_html(full_html=False,include_plotlyjs=False,
                        config=dict(responsive=True,displaylogo=False))
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>原场景与三维语义地图</title><style>
body{margin:24px;font:16px system-ui;color:#172736;background:#fafbfc}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{min-width:0}
img{width:100%;height:570px;object-fit:contain;background:#eef2f5}
.plotly-graph-div{height:650px!important}p{line-height:1.7}.note{max-width:1100px}
@media(max-width:850px){.panels{grid-template-columns:1fr}}
</style><h1>原场景 ↔ 三维语义地图</h1>
<p>拖动右图旋转，滚轮缩放；点击图例可切换物体与桌面二维规划切片。</p>
<script>''' + get_plotlyjs() + '''</script>
<div class="panels"><div class="panel"><h2>IsaacLab 原图</h2><img src="data:image/png;base64,''' + img64 + '''"></div>
<div class="panel"><h2>三维接触语义</h2>''' + graph + '''</div></div>
<p class="note">绿色锤柄：机械臂可接触；红色锤头：保护部位。黄色布娃娃：目标／机械臂可接触；红色杯子和气球：目标／机械臂禁碰。青色半透明点云：目标姿态。</p>
<p class="note">本图使用本次仿真的真实三维碰撞网格、记录位姿和锤子部位点云。它展示三维表面的关系语义；不是已接入规划器的三维体素代价场。桌面目标推动仍查询 (x, y, yaw) 的平面扫掠，全臂避碰查询三维几何。材质为刚体代理。</p>
</html>'''
    (folder/'scene_vs_semantic_3d.html').write_text(html)
    (folder/'semantic_3d_scope.json').write_text(json.dumps(dict(
        mesh_hashes=hashes,footprint_reprojection_pass=True,
        surface_semantics='Per-object relation labels and recorded tool-part point labels',
        planner_scope='Planar target sweep + 3-D whole-arm geometry checks; no volumetric cost planner',
        snapshot_sha256=hashlib.sha256((folder/'map_snapshot.json').read_bytes()).hexdigest()),indent=2))
    print(folder/'scene_vs_semantic_3d.html')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
