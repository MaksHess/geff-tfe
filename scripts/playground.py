# %%
import os

import lsdata

# %%
dataset = lsdata.download('001-mini')
deconv_path = dataset / 'deconv.ome.zarr'
tfe_path = deconv_path / 'tracks' / 'nucleus.tfe'
# %%
os.symlink(deconv_path, tfe_path/'deconv.ome.zarr')