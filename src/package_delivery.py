"""Create a local artifact bundle; does not upload or publish anything."""
import hashlib,json,zipfile
from pathlib import Path
from data import ROOT

def main():
    output=ROOT/'delivery';output.mkdir(exist_ok=True)
    files=[]
    for name in ['.gitignore','.gitattributes','README.md','TOOL_USAGE.md','requirements.txt','ResNet_Codex_Brief_2026.md']:
        files.append(ROOT/name)
    for name in ['configs','src','results','report','runs','submissions']:
        for p in (ROOT/name).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.tmp']:continue
            if p==ROOT/'results/environment.json':continue
            files.append(p)
    files=sorted(set(files))
    manifest=[]
    for p in files:
        relative=p.relative_to(ROOT).as_posix()
        assert not relative.startswith(('.venv/','data/','.git/','delivery/'))
        manifest.append({'path':relative,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    archive=output/'cifar10_resnet_artifacts.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('ARTIFACT_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    result={'archive':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files)+1,'zip_integrity':'passed','includes':'code, configs, report sources/figures, audit/results, all local experiment checkpoints/logs, predictions','excludes':'official image dataset, Python environment, Git history, raw machine process inventory','published':False}
    (output/'package_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
