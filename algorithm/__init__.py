from .standard import standard
from .coteaching import coteaching
from .jocor import jocor

from .loss_ce import ce
from .loss_sce import sce
from .loss_ncerce import ncerce   # apl
from .loss_mae import mae

from .loss_tce import tce
from .loss_ptce import ptce
from .loss_ptceplus import ptceplus

from .loss_gce import gce
from .loss_pgce import pgce
from .loss_pgceplus import pgceplus


from .loss_ael import ael
from .loss_agce import agce
from .loss_aul import aul


__all__ = ('standard',  'coteaching',  'jocor', 
           'ce', 'sce', 'ncerce', 'mae', 
           'tce', 'ptce', 'ptceplus',  
           'gce',  'pgce',  'pgceplus',
           'ael', 'agce', 'aul', 
)