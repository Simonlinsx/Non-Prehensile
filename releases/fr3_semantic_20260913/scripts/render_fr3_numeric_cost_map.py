"""Paper-style RGB -> reviewed instances -> numeric 3-D point contact cost.

This export distinguishes a spatial point query from the full-hammer sweep
query consumed by the recorded controller.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Patch, FancyArrowPatch
import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from scipy.spatial.transform import Rotation
import trimesh

from fr3_semantic_volume import point_contact_cost
from fr3_semantic_cost_map import polygon
from render_fr3_semantic_map import project


def main(folder):
    folder=folder.resolve()
    snapshot=json.loads((folder/'map_snapshot.json').read_text())
    launch=json.loads((folder/'launch.json').read_text())
    volume_runtime=launch['environment'].get('FR3_SEMANTIC_BACKEND')=='volume3d'
    manifest=json.loads((folder/'manifest.jsonl').read_text())
    rgb=np.asarray(Image.open(folder/'scene_original.png'));h,w=rgb.shape[:2]
    camera=snapshot['camera'];objects=[];hashes={}
    for rule in snapshot['obstacles']:
        path=Path(launch['environment']['FR3_SEMANTIC_ASSETS'])/rule['asset_id'].split(':')[0]/'geometry.obj'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert launch['input_hashes'][str(path)]==digest
        hashes[str(path)]=digest
        mesh=trimesh.load(path,force='mesh',process=False)
        spec=next(x for x in manifest['objects'] if x['asset_id']==rule['asset_id'])
        q=np.asarray(rule['quaternion_wxyz'])
        mesh.vertices=Rotation.from_quat(q[[1,2,3,0]]).apply(mesh.vertices*spec['scale'])+rule['position']
        assert np.allclose(polygon(mesh.vertices),rule['polygon'],atol=1e-8)
        objects.append((rule,mesh))
    target=np.asarray(snapshot['target_points'])
    vertices=np.concatenate([m.vertices for _,m in objects]+[target])
    low=vertices.min(0)-[.04,.04,0];high=vertices.max(0)+[.04,.04,.03]
    # Fixed 5 mm spatial sampling. Exact values are calculated at each node;
    # discretized visualization is not a continuous-time safety certificate.
    xs=np.arange(np.floor(low[0]/.005)*.005,high[0]+.005,.005)
    ys=np.arange(np.floor(low[1]/.005)*.005,high[1]+.005,.005)
    zs=np.arange(0,high[2]+.005,.005)
    xx,yy=np.meshgrid(xs,ys)
    cache=folder/'numeric_contact_volume.npz'
    key=hashlib.sha256((folder/'map_snapshot.json').read_bytes()+
        Path(__file__).with_name('fr3_semantic_volume.py').read_bytes()+
        json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    cached=False
    if cache.exists():
        with np.load(cache) as old:
            cached=str(old['query_sha256'])==key
            if cached:soft,hard=old['soft_cost'],old['forbidden']
    if not cached:
        grid=np.concatenate([np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,z)] for z in zs])
        print(f'Computing {len(grid)} 3-D point queries',flush=True)
        value=point_contact_cost(grid,objects)
        soft=value['soft_cost'].reshape(len(zs),len(ys),len(xs))
        hard=value['forbidden'].reshape(soft.shape)
        np.savez_compressed(cache,x_m=xs,y_m=ys,z_m=zs,soft_cost=soft,forbidden=hard,
            total_cost=np.where(hard,np.inf,soft),query_sha256=key)
    maximum=sum(r['weight'] for r,_ in objects if r['mode']=='acceptable')
    norm=Normalize(0,maximum);cmap=plt.get_cmap('viridis');hard_color='#db4281'
    k=int(np.argmin(abs(zs-.015)))
    uv=project(vertices,camera,w,h)
    crop=(max(0,uv[:,0].min()-35),min(w,uv[:,0].max()+35),
          max(0,uv[:,1].min()-35),min(h,uv[:,1].max()+35))

    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12})
    fig=plt.figure(figsize=(17,9),facecolor='white')
    original=fig.add_axes([.025,.48,.21,.39]);annotated=fig.add_axes([.26,.48,.21,.39])
    for ax,title in [(original,'Original scene $I$'),(annotated,'Annotated scene $I\prime$')]:
        ax.imshow(rgb);ax.set_xlim(crop[:2]);ax.set_ylim(crop[3],crop[2]);ax.axis('off')
        ax.set_title(title,fontsize=21,pad=14)
    for i,(rule,mesh) in enumerate(objects,1):
        color=hard_color if rule['mode']=='forbidden' else cmap(norm(rule['weight']))
        # Instance overlay is a projected 3-D mesh silhouette, not a polygon
        # footprint masquerading as segmentation.
        projected=project(mesh.vertices,camera,w,h)
        mask=Image.new('L',(w,h));draw=ImageDraw.Draw(mask)
        for face in mesh.faces:
            draw.polygon([tuple(p) for p in projected[face]],fill=255)
        from matplotlib.colors import to_rgba
        overlay=np.empty((h,w,4));overlay[:]=to_rgba(color)
        overlay[:,:,3]=np.asarray(mask)/255*.42
        annotated.imshow(overlay)
        p=project([mesh.vertices.mean(0)],camera,w,h)[0]
        annotated.text(*p,str(i),ha='center',va='center',fontsize=20,
            bbox=dict(boxstyle='round,pad=.2',fc='white',ec='black',lw=1.3))
    vlm=fig.add_axes([.485,.43,.105,.24]);vlm.axis('off')
    vlm.text(.5,.5,'Vision\nLanguage\nModel',ha='center',va='center',fontsize=20,
        color='#164675',weight='bold',bbox=dict(boxstyle='round,pad=.55',fc='#d9eaf5',ec='black',lw=1.5))
    fig.text(.537,.70,'Assistant review\nfor this prototype',ha='center',fontsize=11)
    instruction=fig.add_axes([.04,.16,.40,.23]);instruction.axis('off')
    instruction.text(.5,.55,'Safely extract the hammer.\n(1) Doll: contact is acceptable.\n(2) Liquid cup and (3) balloon: avoid contact.',
        ha='center',va='center',fontsize=16,linespacing=1.5,
        bbox=dict(boxstyle='round,pad=.6',fc='#e5f0f7',ec='black',ls='--',lw=1.5))
    instruction.text(.5,-.05,'Language instruction $\ell$',ha='center',fontsize=20)
    for a,b in [((.236,.66),(.256,.66)),((.47,.66),(.492,.60)),((.44,.28),(.518,.43)),((.59,.57),(.646,.72))]:
        fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',
            mutation_scale=20,linewidth=1.5,color='#6088a0',connectionstyle='arc3,rad=0'))
    ax3=fig.add_axes([.64,.54,.32,.37],projection='3d')
    ax3.set_title('3-D contact cost $C(p)$',fontsize=20,pad=10)
    np.random.seed(0)  # Only offline display sampling, no rollout state.
    for rule,mesh in objects:
        v,_=trimesh.sample.sample_surface(mesh,3500)
        color=hard_color if rule['mode']=='forbidden' else cmap(norm(rule['weight']))
        ax3.scatter(v[:,0],v[:,1],v[:,2],c=[color],s=1.5,depthshade=True)
        p=v.mean(0);p[2]=v[:,2].max()+.012
        if rule['mode']=='acceptable':
            anchor=p.copy();p+=np.array([-.045,0,.065])
            ax3.plot([anchor[0],p[0]],[anchor[1],p[1]],[anchor[2],p[2]],color='#38424a',lw=1)
        label='$\infty$ (forbidden)' if rule['mode']=='forbidden' else f'{rule["weight"]:.2f}'
        ax3.text(*p,label,fontsize=11,ha='center',bbox=dict(fc='white',ec='none',alpha=.85))
    ax3.scatter(target[:,0],target[:,1],target[:,2],c='#6d7b8a',s=.5,alpha=.2)
    ax3.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',zlim=(0,high[2]+.025))
    ax3.set_box_aspect((high-low)*[1,1,1.4]);ax3.view_init(elev=29,azim=-61)
    ax3.tick_params(labelsize=9,pad=0)
    ax2=fig.add_axes([.66,.12,.28,.33])
    image=ax2.pcolormesh(xs,ys,np.ma.masked_where(hard[k],soft[k]),cmap=cmap,norm=norm,shading='nearest',rasterized=True)
    ax2.contourf(xx,yy,hard[k].astype(float),levels=[.5,1.5],colors=[hard_color],hatches=['///'])
    ax2.scatter(target[:,0],target[:,1],c='white',s=.5,alpha=.45)
    for i,(rule,mesh) in enumerate(objects,1):
        p=mesh.vertices.mean(0)
        ax2.text(p[0],p[1],str(i),ha='center',va='center',fontsize=11,
            bbox=dict(fc='white',ec='none',alpha=.85,pad=2))
    ax2.set(xlabel='X (m)',ylabel='Y (m)',aspect='equal')
    ax2.set_title(f'XY slice at Z = {zs[k]:.3f} m',fontsize=17,pad=10)
    colorax=fig.add_axes([.949,.15,.014,.27]);bar=fig.colorbar(image,cax=colorax)
    bar.set_label('Finite contact cost (dimensionless)',fontsize=11)
    bar.set_ticks(np.linspace(0,maximum,4))
    fig.legend(handles=[Patch(facecolor=hard_color,hatch='///',label='Forbidden: $C=\infty$ (separate from finite color scale)')],
        loc='lower right',bbox_to_anchor=(.976,.015),frameon=False,fontsize=11)
    fig.text(.035,.06,'3-D point query: 5 mm grid; 12 mm clearance band.\nGeometry: recorded simulation meshes and poses.',fontsize=11,color='#354354')
    fig.savefig(folder/'semantic_cost_pipeline.png',dpi=180,bbox_inches='tight')
    fig.savefig(folder/'semantic_cost_pipeline.pdf',bbox_inches='tight')
    plt.close(fig)

    # Interactive quantitative map: the slider changes real Z, not opacity.
    plot=go.Figure()
    def surfaces(index):
        z=np.full_like(xx,zs[index])
        return [go.Surface(x=xx,y=yy,z=z,surfacecolor=np.where(hard[index],np.nan,soft[index]),
            cmin=0,cmax=maximum,colorscale='Viridis',connectgaps=False,
            colorbar=dict(title='Finite cost',tickvals=np.linspace(0,maximum,4),len=.65),
            name='Finite contact cost',hovertemplate='X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m<br>Cost=%{surfacecolor:.3f}<extra></extra>'),
            go.Surface(x=xx,y=yy,z=np.where(hard[index],z,np.nan),surfacecolor=np.ones_like(xx),
            colorscale=[[0,hard_color],[1,hard_color]],showscale=False,connectgaps=False,
            name='Forbidden (infinite cost)',hovertemplate='X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m<br>Forbidden: infinite cost<extra></extra>')]
    for trace in surfaces(k):plot.add_trace(trace)
    for rule,mesh in objects:
        v=mesh.vertices;f=mesh.faces
        plot.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
            color=hard_color if rule['mode']=='forbidden' else '#f5d53d',opacity=.18,
            name=rule['id'],showlegend=True,hoverinfo='name'))
    plot.frames=[go.Frame(name=str(i),data=surfaces(i),traces=[0,1]) for i in range(len(zs))]
    plot.update_layout(margin=dict(l=0,r=35,t=10,b=0),paper_bgcolor='white',
        scene=dict(aspectmode='data',xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)',
            zaxis=dict(range=[0,high[2]+.01])),legend=dict(orientation='h',y=-.05),
        sliders=[dict(active=k,currentvalue=dict(prefix='Height Z = ',suffix=' m'),pad=dict(t=20),
            steps=[dict(label=f'{z:.3f}',method='animate',args=[[str(i)],dict(mode='immediate',
                frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i,z in enumerate(zs)])])
    graph=plot.to_html(full_html=False,include_plotlyjs=False,config=dict(responsive=True,displaylogo=False))
    encoded=base64.b64encode((folder/'scene_original.png').read_bytes()).decode()
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>三维数值语义代价图</title>
<style>body{font:16px system-ui;margin:24px;color:#182c3b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid>div{min-width:0}img{width:100%;height:570px;object-fit:contain}.plotly-graph-div{height:650px!important}p{line-height:1.7}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<h1>原场景与三维数值代价图</h1><p>X/Y/Z 单位为米；拖动旋转，移动滑条查看不同高度。鼠标悬停读取坐标和代价。紫红色为禁碰区域，代价为 ∞。</p><script>'''+get_plotlyjs()+'''</script>
<div class="grid"><div><h2>IsaacLab 原图</h2><img src="data:image/png;base64,'''+encoded+'''"></div><div><h2>三维代价场的高度切片</h2>'''+graph+'''</div></div>
<p>令 dᵢ(p) 为三维点 p 到物体语义体积的距离（体积内为 0），δ=0.012 m。禁碰物体采用保守三维凸包，杯内空腔也禁入；布娃娃采用实际碰撞部件体积的并集。若任一禁碰物体满足 dᵢ(p)≤δ，则 C(p)=∞；否则 C(p)=Σᵢ wᵢ max(0,1−dᵢ(p)/δ)，仅对允许接触物体求和。本场布娃娃 w=0.15，因此有限值范围为 0–0.15，0.075 对应距其表面 6 mm。权重是指定的策略参数，不是概率或实测受损程度。</p>
<p>RUNTIME_SCOPE_TEXT 锤子部位 C1 与全臂几何检查仍独立执行。刚体代理不证明真实软体／液体安全。</p></html>'''
    runtime_scope=('本图显示三维点接触代价；本次执行使用同一语义规则下的整把锤子三维扫掠查询。控制器在线查询几何和当前位姿，不读取本图的离线栅格。'
        if volume_runtime else '本图是三维点接触代价查询的新可视化，不是整把锤子的位姿代价；它尚未替换该冻结仿真中的二维目标扫掠检查。')
    html=html.replace('RUNTIME_SCOPE_TEXT',runtime_scope)
    (folder/'numeric_cost_map_3d.html').write_text(html)
    scope=dict(grid_spacing_m=.005,margin_m=.012,finite_range=[0,maximum],hard_cost='infinity',
        formula='If any forbidden d_i<=delta: infinity; else sum acceptable w_i*max(0,1-d_i/delta)',
        distance='Euclidean distance to semantic volume: forbidden object convex envelope (including cup cavity), acceptable union of actual convex collision components; zero inside; exact closest triangles outside',
        query='3-D point contact, not full hammer pose or robot configuration',
        consumed_by_successful_rollouts=False,scene_sha256=key,mesh_hashes=hashes,
        runtime_backend='volume3d' if volume_runtime else 'planar',runtime_scope=runtime_scope,
        limitation='Offline spatial field visualization added after frozen trials; not autonomous VLM inference or continuous safety proof')
    (folder/'numeric_cost_scope.json').write_text(json.dumps(scope,indent=2))
    print(folder/'semantic_cost_pipeline.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
