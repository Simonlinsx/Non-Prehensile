"""Same-camera Isaac RGB / consumed semantic cost-map comparison artifact."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon, Patch
from fr3_semantic_cost_map import polygon, query


def project(points,camera,width,height):
    p=np.asarray(points)
    projection=np.array(camera['projection'],dtype=float,copy=True)
    # The viewport USD camera carries the interactive viewport's aperture
    # aspect; env.render() uses a separate 1280x720 render product. Preserve
    # horizontal FOV and conform vertical FOV to the actual image resolution.
    projection[1,1]=projection[0,0]*width/height
    clip=np.c_[p,np.ones(len(p))]@np.asarray(camera['view'])@projection
    ndc=clip[:,:2]/clip[:,3,None]
    return np.c_[(ndc[:,0]+1)*width/2,(1-ndc[:,1])*height/2]


def main(folder):
    folder=folder.resolve();d=json.loads((folder/'map_snapshot.json').read_text())
    rgb=np.asarray(Image.open(folder/'scene_original.png'));h,w=rgb.shape[:2];cam=d['camera']
    pts=np.asarray(d['target_points']);foot=np.asarray(d.get('target_footprint',polygon(pts)));center=np.asarray(d['target_center'])
    obstacles=d['obstacles'];objects=np.concatenate([np.asarray(o['polygon']) for o in obstacles]+[foot])
    pose=np.asarray(d['target_pose']);goal=np.asarray(d['goal'])
    current_rotation=Rotation.from_quat(pose[[4,5,6,3]])
    goal_rotation=Rotation.from_quat(goal[[4,5,6,3]])
    goal_center=goal_rotation.apply(current_rotation.inv().apply(center-pose[:3]))+goal[:3]
    goal_points=goal_rotation.apply(current_rotation.inv().apply(pts-pose[:3]))+goal[:3]
    goal_footprint=polygon(goal_points)
    lo=objects.min(0)-.065;hi=objects.max(0)+.065
    xs=np.linspace(lo[0],hi[0],95);ys=np.linspace(lo[1],hi[1],95)
    xx,yy=np.meshgrid(xs,ys)
    grid=np.c_[xx.reshape(-1),yy.reshape(-1)]
    uv=project(np.c_[grid,np.full(len(grid),.0005)],cam,w,h).reshape(*xx.shape,2)
    data=[]
    for degrees in (0,30,60):
        theta=np.deg2rad(degrees);c,s=np.cos(theta),np.sin(theta)
        shape=(foot-center[:2])@np.array([[c,s],[-s,c]])+center[:2]
        values=[];soft=[]
        for xy in grid:
            # Static pose slice, not a swept trajectory from the current pose.
            result=query(shape+xy-center[:2],xy,[0,0],0,obstacles)
            values.append(2 if not result['admissible'] else 1 if result['soft_cost']>0 else 0)
            soft.append(result['soft_cost'])
        data.append((degrees,np.array(values).reshape(xx.shape),np.array(soft).reshape(xx.shape)))
    np.savez_compressed(folder/'cost_map_slices.npz',x=xs,y=ys,classes=np.array([x[1] for x in data]),soft_cost=np.array([x[2] for x in data]),yaw_offsets_deg=[0,30,60])
    cmap=ListedColormap(['#80cc9e','#f3c64d','#ec7779']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    all_uv=project(np.c_[objects,np.zeros(len(objects))],cam,w,h)
    # Include object height for crop, so the cup rim / balloon remain visible.
    upper=project(np.c_[objects,np.full(len(objects),.10)],cam,w,h)
    all_uv=np.r_[all_uv,upper]
    xmin=max(0,all_uv[:,0].min()-65);xmax=min(w,all_uv[:,0].max()+65)
    ymin=max(0,all_uv[:,1].min()-55);ymax=min(h,all_uv[:,1].max()+45)
    def draw(ax,map_data=None):
        ax.imshow(rgb,alpha=1 if map_data is None else .36)
        if map_data is None:
            for i,o in enumerate(obstacles,1):
                p=project([o['position']],cam,w,h)[0]
                ax.annotate(str(i),p,xytext=(0,15),textcoords='offset points',ha='center',fontsize=10,weight='bold',color='#172736',bbox=dict(boxstyle='circle,pad=.2',fc='white',ec='#172736',lw=.7,alpha=.95))
        if map_data is not None:
            ax.pcolormesh(uv[:,:,0],uv[:,:,1],map_data,cmap=cmap,norm=norm,shading='nearest',alpha=.87,zorder=2,rasterized=True)
            for i,o in enumerate(obstacles,1):
                poly=np.asarray(o['polygon']);pu=project(np.c_[poly,np.full(len(poly),.002)],cam,w,h)
                ax.add_patch(Polygon(pu,closed=True,facecolor='#ffda55' if o['mode']=='acceptable' else '#cc2438',edgecolor='white',linewidth=1.5,zorder=3))
                p=project([o['position']],cam,w,h)[0]
                ax.text(*p,f'{i}  {o["id"]}',fontsize=10,weight='bold',ha='center',color='#192532',bbox=dict(facecolor='white',alpha=.92,edgecolor='none',pad=3),zorder=5)
            t=project(np.c_[foot,np.full(len(foot),.005)],cam,w,h)
            ax.add_patch(Polygon(t,closed=True,facecolor='none',edgecolor='#172736',linewidth=2,zorder=4))
            if 'target_semantics' in d:
                semantics=np.asarray(d['target_semantics']);pu=project(pts,cam,w,h)
                protected=semantics[:,1]>=.25
                safe=(semantics[:,0]>=.25)&~protected
                ax.scatter(pu[safe,0],pu[safe,1],c='#17844c',s=3,linewidths=0,zorder=4)
                ax.scatter(pu[protected,0],pu[protected,1],c='#b91432',s=3,linewidths=0,zorder=4)
            goal_uv=project(np.c_[goal_footprint,np.full(len(goal_footprint),.005)],cam,w,h)
            ax.add_patch(Polygon(goal_uv,closed=True,facecolor='none',edgecolor='#008da5',linestyle='--',linewidth=1.5,zorder=4))
            c=project([center],cam,w,h)[0];g=project([goal_center],cam,w,h)[0]
            ax.scatter(*c,c='#18283a',s=45,zorder=6);ax.scatter(*g,c='#06b6cf',marker='*',s=160,edgecolors='#153644',zorder=6)
            ax.annotate('Goal',g,xytext=(9,8),textcoords='offset points',color='#004754',weight='bold',fontsize=10,zorder=6)
        ax.set_xlim(xmin,xmax);ax.set_ylim(ymax,ymin);ax.axis('off')
    fig,axs=plt.subplots(1,2,figsize=(14,4.5),facecolor='#fafbfc')
    fig.subplots_adjust(left=.012,right=.99,top=.78,bottom=.17,wspace=.025)
    draw(axs[0]);draw(axs[1],data[0][1])
    axs[0].set_title('Original IsaacLab scene',loc='left',fontsize=16,weight='bold',pad=12)
    axs[1].set_title('Semantic cost map | current hammer orientation',loc='left',fontsize=14,weight='bold',pad=12)
    fig.legend(handles=[Patch(color='#80cc9e',label='No semantic penalty'),Patch(color='#f3c64d',label='Acceptable contact: finite cost'),Patch(color='#ec7779',label='Forbidden: hard exclusion')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.04))
    fig.text(.5,.02,'Hammer surface: green = legal robot contact; red = protected part. Dashed cyan outline = goal pose.',ha='center',fontsize=9,color='#354354')
    fig.suptitle('Semantic contact cost for hammer placement',fontsize=18,weight='bold')
    fig.savefig(folder/'scene_vs_cost_map.png',dpi=180,bbox_inches='tight');fig.savefig(folder/'scene_vs_cost_map.pdf',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,(degrees,values,_) in zip(axs,data):draw(ax,values);ax.set_title(f'Hammer orientation: current + {degrees} deg')
    fig.suptitle('Orientation-conditioned pose costs | same geometry and policy',weight='bold')
    fig.savefig(folder/'orientation_cost_maps.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (folder/'map_visualization_scope.json').write_text(json.dumps(dict(query='Same polygon SAT query function used by candidate/micro guards. Pose slices evaluate zero displacement, not path feasibility. Red region inflated by full hammer footprint plus 12 mm; it is not an object segmentation mask.',projection='USD camera view/projection matrices; same perspective as original RGB.',scope='Geometry-based semantic cost only; free map cells do not guarantee arm reachability, C1 contact availability or task success.'),indent=2))
    print(folder/'scene_vs_cost_map.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);main(p.parse_args().folder)
