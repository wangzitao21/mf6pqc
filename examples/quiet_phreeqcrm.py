"""Optional case-level factories that do not create PhreeqcRM text files.

Keep ChemistryOptions(print_chemistry_mask=0).  Replace a case's factory with
QuietProcessBackendFactory(processes=...) or QuietNativeBackendFactory().
This skips OpenFiles only; selected-output arrays and MF6PQC results are kept.
"""

from mf6pqc.backends import NativeBackendFactory
from mf6pqc.parallel import ProcessBackendFactory


def _skip_open_files():
    return 0


class QuietNativeBackendFactory(NativeBackendFactory):
    """Keep native chemistry results without opening .chem.txt/.log.txt."""

    def create_phreeqcrm(self, nxyz, nthreads):
        chemistry = super().create_phreeqcrm(nxyz, nthreads)
        chemistry.OpenFiles = _skip_open_files
        return chemistry


class QuietProcessBackendFactory(ProcessBackendFactory):
    """Keep process parallelism without opening per-worker text files."""

    def create_phreeqcrm(self, nxyz, nthreads):
        chemistry = super().create_phreeqcrm(nxyz, nthreads)
        chemistry.OpenFiles = _skip_open_files
        return chemistry
