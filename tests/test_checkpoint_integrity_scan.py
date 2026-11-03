import json
from pathlib import Path
import struct
import zipfile

from scripts import check_fedsharp_checkpoint_integrity as scan


def archive(path):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as output:
        for name in sorted(scan.EXPECTED):output.writestr(name,b'test-content-'+name.encode())


def corrupt(path,name):
    with zipfile.ZipFile(path) as source:
        item=source.getinfo(name);offset=item.header_offset
    with path.open('r+b') as file:
        file.seek(offset);header=file.read(30)
        length,extra=struct.unpack_from('<HH',header,26);file.seek(offset+30+length+extra)
        byte=file.read(1);file.seek(-1,1);file.write(bytes([byte[0]^1]))


def test_crc_scan_checks_all_members_and_keeps_bytes(tmp_path):
    path=tmp_path/'checkpoint.npz';archive(path)
    assert scan.inspect(path)['ok']
    corrupt(path,'outbound.npy');before=path.read_bytes()
    row=scan.inspect(path)
    assert not row['ok'] and row['errors']==['outbound.npy']
    assert row['members']['nodeParams.npy']=='CRC_OK'
    assert row['members']['metadata.npy']=='CRC_OK'
    assert row['fileUnchanged'] and path.read_bytes()==before


def test_scan_repeats_only_bad_local_files_and_never_opens_remote_paths(tmp_path,monkeypatch):
    state=dict(runs={})
    for i in range(3):
        root=tmp_path/f'output{i}';folder=root/f'run{i}';folder.mkdir(parents=True)
        path=folder/'checkpoint.npz';archive(path)
        if i==0:corrupt(path,'outbound.npy')
        state['runs'][f'run{i}']=dict(owner='hal9k' if i<2 else 'themachine',location=dict(output=str(root)))
    folder=tmp_path/'fleet';folder.mkdir();(folder/'state.json').write_text(json.dumps(state))
    monkeypatch.setattr(scan.socket,'gethostname',lambda:'hal9k')
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=scan.scan(tmp_path,repeats=2,progress=lambda *a,**kw:None)
    assert result['checked']==2 and result['initiallyPassed']==1 and len(result['badCheckpoints'])==1
    bad=result['badCheckpoints'][0]
    assert all(v['stableChecksum'] and not v['archive']['ok'] for v in bad['repeatReads'])
    assert all(p.read_bytes()==value for p,value in before.items())
    assert {p for p in tmp_path.rglob('*') if p.is_file()}==set(before)


def test_bad_header_or_missing_member_is_reported(tmp_path):
    path=tmp_path/'bad.npz';path.write_bytes(b'broken archive')
    assert not scan.inspect(path)['ok']
    path=tmp_path/'missing.npz'
    with zipfile.ZipFile(path,'w') as output:output.writestr('metadata.npy',b'metadata')
    assert not scan.inspect(path)['ok']
