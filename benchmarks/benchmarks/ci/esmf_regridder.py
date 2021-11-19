"""Quick running benchmarks for :mod:`esmf_regrid.esmf_regridder`."""

import os
from pathlib import Path

import numpy as np
import dask.array as da
import iris
from iris.coord_systems import RotatedGeogCS
from iris.cube import Cube

from benchmarks import disable_repeat_between_setup
from esmf_regrid.esmf_regridder import GridInfo
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.schemes import ESMFAreaWeightedRegridder

from ..generate_data import _grid_cube


def _make_small_grid_args():
    """
    Not importing the one in test_GridInfo - if that changes, these benchmarks
    would 'invisibly' change too.

    """
    small_x = 2
    small_y = 3
    small_grid_lon = np.array(range(small_x)) / (small_x + 1)
    small_grid_lat = np.array(range(small_y)) * 2 / (small_y + 1)

    small_grid_lon_bounds = np.array(range(small_x + 1)) / (small_x + 1)
    small_grid_lat_bounds = np.array(range(small_y + 1)) * 2 / (small_y + 1)
    return (
        small_grid_lon,
        small_grid_lat,
        small_grid_lon_bounds,
        small_grid_lat_bounds,
    )


class TimeGridInfo:
    def setup(self):
        lon, lat, lon_bounds, lat_bounds = _make_small_grid_args()
        self.grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    def time_make_grid(self):
        """Basic test for :meth:`~esmf_regrid.esmf_regridder.GridInfo.make_esmf_field`."""
        esmf_grid = self.grid.make_esmf_field()
        esmf_grid.data[:] = 0

    time_make_grid.version = 1


class MultiGridCompare:
    params = ["similar", "large_source", "large_target", "mixed"]
    param_names = ["source/target difference"]

    def get_args(self, type):
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 20
        n_lats_src = 40
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 100
        if type == "large_source":
            n_lons_src = 100
            n_lats_src = 200
        if type == "large_target":
            n_lons_tgt = 100
            n_lats_tgt = 200
        alt_coord_system = type == "mixed"
        args = (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            h,
            coord_system_src,
        )
        return args


class TimeRegridding(MultiGridCompare):
    def setup(self, type):
        (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            h,
            coord_system_src,
        ) = self.get_args(type)
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            alt_coord_system=alt_coord_system,
        )
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape(
            [n_lats_src, n_lons_src, h]
        )
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        self.regrid_class = ESMFAreaWeightedRegridder
        self.regridder = self.regrid_class(src, tgt)
        self.src = src
        self.tgt = tgt

    def time_prepare_regridding(self, type):
        _ = self.regrid_class(self.src, self.tgt)

    def time_perform_regridding(self, type):
        _ = self.regridder(self.src)


@disable_repeat_between_setup
class TimeLazyRegridding:
    def setup_cache(self):
        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)
        file = str(SYNTH_DATA_DIR.joinpath("chunked_cube.nc"))
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 100
        n_lats_src = 200
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 2000
        # Rotated coord systems prevent pickling of the regridder so are
        # removed for the time being.
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            # alt_coord_system=True,
        )
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)

        chunk_size = [n_lats_src, n_lons_src, 10]
        src_data = da.ones([n_lats_src, n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        iris.save(src, file, chunksizes=chunk_size)
        # Construct regridder with a loaded version of the grid for consistency.
        loaded_src = iris.load_cube(file)
        regridder = ESMFAreaWeightedRegridder(loaded_src, tgt)

        return regridder, file

    def setup(self, cache):
        regridder, file = cache
        self.src = iris.load_cube(file)
        cube = iris.load_cube(file)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        assert self.src.has_lazy_data()
        regridder, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        assert self.result.has_lazy_data()
        _ = self.result.data


class TimeMeshToGridRegridding(TimeRegridding):
    def setup(self, type):
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            h,
            alt_coord_system_src,
        ) = self.get_args(type)
        mesh = _gridlike_mesh(n_lons_src, n_lats_src)
        tgt = _grid_cube(
            n_lons_tgt,
            n_lats_tgt,
            lon_bounds,
            lat_bounds,
            alt_coord_system=alt_coord_system_src,
        )
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape([-1, h])
        src = Cube(src_data)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        src.add_aux_coord(mesh_coord_x, 0)
        src.add_aux_coord(mesh_coord_y, 0)
        self.regrid_class = MeshToGridESMFRegridder
        self.regridder = self.regrid_class(src, tgt)
        self.src = src
        self.tgt = tgt


@disable_repeat_between_setup
class TimeLazyMeshToGridRegridding:
    def setup_cache(self):
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)
        file = str(SYNTH_DATA_DIR.joinpath("chunked_cube.nc"))
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 100
        n_lats_src = 200
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 2000
        mesh = _gridlike_mesh(n_lons_src, n_lats_src)
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)

        chunk_size = [n_lats_src * n_lons_src, 10]
        src_data = da.ones([n_lats_src * n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        iris.save(src, file, chunksizes=chunk_size)
        # Construct regridder with a loaded version of the grid for consistency.
        loaded_src = iris.load_cube(file)
        # While iris is not able to save meshes, we add these after loading.
        # TODO: change this back after iris allows mesh saving.
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        loaded_src.add_aux_coord(mesh_coord_x, 0)
        loaded_src.add_aux_coord(mesh_coord_y, 0)
        regridder = MeshToGridESMFRegridder(loaded_src, tgt)

        return regridder, file, mesh

    def setup(self, cache):
        regridder, file, mesh = cache
        self.src = iris.load_cube(file)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        self.src.add_aux_coord(mesh_coord_x, 0)
        self.src.add_aux_coord(mesh_coord_y, 0)
        cube = iris.load_cube(file)
        # While iris is not able to save meshes, we add these after loading.
        # TODO: change this back after iris allows mesh saving.
        cube.add_aux_coord(mesh_coord_x, 0)
        cube.add_aux_coord(mesh_coord_y, 0)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        assert self.src.has_lazy_data()
        regridder, _, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        assert self.result.has_lazy_data()
        _ = self.result.data


class TimeGridToMeshRegridding(TimeRegridding):
    def setup(self, type):
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            h,
            coord_system_src,
        ) = self.get_args(type)
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            coord_system=coord_system_src,
        )
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape(
            [n_lats_src, n_lons_src, h]
        )
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        tgt_data = np.zeros(n_lats_tgt * n_lons_tgt)
        tgt = Cube(tgt_data)
        mesh = _gridlike_mesh(n_lons_tgt, n_lats_tgt)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        tgt.add_aux_coord(mesh_coord_x, 0)
        tgt.add_aux_coord(mesh_coord_y, 0)
        self.regrid_class = GridToMeshESMFRegridder
        self.regridder = self.regrid_class(src, tgt)
        self.src = src
        self.tgt = tgt


@disable_repeat_between_setup
class TimeLazyGridToMeshRegridding:
    def setup_cache(self):
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)
        file = str(SYNTH_DATA_DIR.joinpath("chunked_cube.nc"))
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 100
        n_lats_src = 200
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 2000
        mesh = _gridlike_mesh(n_lons_tgt, n_lats_tgt)
        grid = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds)

        chunk_size = [n_lats_src, n_lons_src, 10]
        src_data = da.ones([n_lats_src, n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        tgt_data = np.zeros(n_lats_tgt * n_lons_tgt)
        tgt = Cube(tgt_data)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        tgt.add_aux_coord(mesh_coord_x, 0)
        tgt.add_aux_coord(mesh_coord_y, 0)
        iris.save(src, file, chunksizes=chunk_size)
        # Construct regridder with a loaded version of the grid for consistency.
        loaded_src = iris.load_cube(file)
        regridder = GridToMeshESMFRegridder(loaded_src, tgt)

        return regridder, file

    def setup(self, cache):
        regridder, file = cache
        self.src = iris.load_cube(file)
        cube = iris.load_cube(file)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        assert self.src.has_lazy_data()
        regridder, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        assert self.result.has_lazy_data()
        _ = self.result.data


class TimeRegridderIO(MultiGridCompare):
    def setup(self, type):
        from esmf_regrid.experimental.io import load_regridder, save_regridder

        self.load_regridder = load_regridder
        self.save_regridder = save_regridder
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            _,
            alt_coord_system_src,
        ) = self.get_args(type)
        src_grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            alt_coord_system=alt_coord_system_src,
        )
        tgt_grid = _grid_cube(
            n_lons_tgt,
            n_lats_tgt,
            lon_bounds,
            lat_bounds,
            coord_system=coord_system_src,
        )
        src_mesh = _gridlike_mesh(
            n_lons_src,
            n_lats_src,
        )
        tgt_mesh = _gridlike_mesh(
            n_lons_tgt,
            n_lats_tgt,
        )

        def mesh_to_cube(mesh, data):
            cube = Cube(data)
            mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
            cube.add_aux_coord(mesh_coord_x, 0)
            cube.add_aux_coord(mesh_coord_y, 0)
            return cube

        src_data = np.zeros([n_lats_src * n_lons_src])
        src_mesh_cube = mesh_to_cube(src_mesh, src_data)
        tgt_data = np.zeros([n_lats_tgt * n_lons_tgt])
        tgt_mesh_cube = mesh_to_cube(tgt_mesh, tgt_data)
        self.mesh_to_grid_regridder = MeshToGridESMFRegridder(src_mesh_cube, tgt_grid)
        self.grid_to_mesh_regridder = GridToMeshESMFRegridder(src_grid, tgt_mesh_cube)

        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)
        self.source_file_m2g = str(SYNTH_DATA_DIR.joinpath(f"m2g_source_{type}.nc"))
        self.source_file_g2m = str(SYNTH_DATA_DIR.joinpath(f"g2m_source_{type}.nc"))
        self.destination_file_m2g = str(
            SYNTH_DATA_DIR.joinpath(f"m2g_destination_{type}.nc")
        )
        self.destination_file_g2m = str(
            SYNTH_DATA_DIR.joinpath(f"g2m_destination_{type}.nc")
        )
        save_regridder(self.mesh_to_grid_regridder, self.source_file_m2g)
        save_regridder(self.grid_to_mesh_regridder, self.source_file_g2m)

    def teardown(self, type):
        if os.path.exists(self.source_file_m2g):
            os.remove(self.source_file_m2g)
        if os.path.exists(self.source_file_g2m):
            os.remove(self.source_file_g2m)
        if os.path.exists(self.destination_file_m2g):
            os.remove(self.destination_file_m2g)
        if os.path.exists(self.destination_file_g2m):
            os.remove(self.destination_file_g2m)

    def time_save_mesh_to_grid(self, type):
        self.save_regridder(self.mesh_to_grid_regridder, self.destination_file_m2g)

    def time_save_grid_to_mesh(self, type):
        self.save_regridder(self.grid_to_mesh_regridder, self.destination_file_g2m)

    def time_load_mesh_to_grid(self, type):
        _ = self.load_regridder(self.source_file_m2g)

    def time_load_grid_to_mesh(self, type):
        _ = self.load_regridder(self.source_file_g2m)
