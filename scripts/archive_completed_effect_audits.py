#!/usr/bin/env python3
"""Losslessly archive completed debug effect logs, with checked restoration."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time


ROOT=Path(__file__).resolve().parents[1]/'outputs/contact_planner_m3'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_scene(path):
    scene=path.resolve()
    if not scene.is_relative_to(ROOT.resolve()):
        raise ValueError('Only this project\'s contact-planner outputs can be archived')
    result=scene/'result.json'
    r=json.loads(result.read_text())
    if (r.get('schema')!='nonprehensile.c3_online_isaaclab_task.v1'
            or not r.get('executed_steps') or not r.get('stopped_reason')
            or time.time()-result.stat().st_mtime<600):
        raise ValueError('Requires a completed result older than ten minutes')
    return scene


def archive(scene):
    scene=validate_scene(scene)
    original=scene/'effect_audit.jsonl'
    compressed=scene/'effect_audit.jsonl.gz'
    metadata=scene/'effect_audit_archive.json'
    temporary=scene/'effect_audit.jsonl.gz.partial'
    if compressed.exists() or metadata.exists():
        raise FileExistsError('Archive already exists; original evidence will not be overwritten')
    before=original.stat()
    h=hashlib.sha256()
    with original.open('rb') as source, temporary.open('xb') as target:
        with gzip.GzipFile(filename='',mode='wb',fileobj=target,compresslevel=6,mtime=0) as stream:
            for chunk in iter(lambda:source.read(1024*1024),b''):
                h.update(chunk);stream.write(chunk)
    check=hashlib.sha256()
    with gzip.open(temporary,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):check.update(chunk)
    after=original.stat()
    if (h.digest()!=check.digest() or (before.st_ino,before.st_size,before.st_mtime_ns)
            !=(after.st_ino,after.st_size,after.st_mtime_ns)):
        raise ValueError('Archive verification failed or source changed; original retained')
    temporary.rename(compressed)
    report=dict(schema='nonprehensile.effect_audit_archive.v1',
                source=str(original),archive=str(compressed),source_bytes=before.st_size,
                archive_bytes=compressed.stat().st_size,source_sha256=h.hexdigest(),
                archive_sha256=digest(compressed),verified_lossless=True,
                restore_argv=['python',str(Path(__file__).resolve()),'restore','--metadata',str(metadata)])
    metadata.write_text(json.dumps(report,indent=2)+'\n')
    original.unlink()
    return report


def restore(metadata):
    meta=json.loads(metadata.read_text())
    original=Path(meta['source']);compressed=Path(meta['archive'])
    if (original.name!='effect_audit.jsonl' or compressed.name!='effect_audit.jsonl.gz'
            or original.parent.resolve()!=metadata.parent.resolve()
            or compressed.parent.resolve()!=metadata.parent.resolve()
            or not original.resolve().is_relative_to(ROOT.resolve())):
        raise ValueError('Archive paths must stay in their recorded project scene')
    if digest(compressed)!=meta['archive_sha256']:
        raise ValueError('Compressed evidence changed')
    if original.exists():
        if digest(original)!=meta['source_sha256']:
            raise ValueError('Existing original differs; refusing overwrite')
        return dict(restored=True,already_present=True,source=str(original))
    temporary=original.with_name(original.name+'.restore.partial')
    h=hashlib.sha256()
    with gzip.open(compressed,'rb') as source,temporary.open('xb') as target:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            h.update(chunk);target.write(chunk)
    if h.hexdigest()!=meta['source_sha256'] or temporary.stat().st_size!=meta['source_bytes']:
        raise ValueError('Restored bytes failed verification')
    temporary.rename(original)
    return dict(restored=True,source=str(original),source_sha256=h.hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('archive');p.add_argument('--scene-directory',type=Path,action='append',required=True)
    p=sub.add_parser('restore');p.add_argument('--metadata',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='archive':
        for path in args.scene_directory:print(json.dumps(archive(path)),flush=True)
    else: print(json.dumps(restore(args.metadata)))


if __name__=='__main__':main()
