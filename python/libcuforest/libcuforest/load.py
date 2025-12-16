# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import ctypes
import os

# Loading with RTLD_LOCAL adds the library itself to the loader's
# loaded library cache without loading any symbols into the global
# namespace. This allows libraries that express a dependency on
# this library to be loaded later and successfully satisfy this dependency
# without polluting the global symbol table with symbols from
# libcuforest that could conflict with symbols from other DSOs.
PREFERRED_LOAD_FLAG = ctypes.RTLD_LOCAL


def _load_system_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``
    Raises ``OSError`` if library cannot be loaded.
    """
    return ctypes.CDLL(soname, PREFERRED_LOAD_FLAG)


def _load_wheel_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``

    Returns ``None`` if the library cannot be loaded.
    """
    if os.path.isfile(lib := os.path.join(os.path.dirname(__file__), "lib64", soname)):
        return ctypes.CDLL(lib, PREFERRED_LOAD_FLAG)
    return None


def load_library():
    """Dynamically load libcuforest++.so and its dependencies"""
    try:
        # These libraries must all be loaded before libcuforest
        import libraft
        import librmm
        import rapids_logger

        rapids_logger.load_library()
        librmm.load_library()
        libraft.load_library()
    except ModuleNotFoundError:
        # These runtime dependencies might be satisfied by conda packages (which do not
        # have any Python modules) instead of wheels. In that situation, assume that
        # the libraries are in a place where the loader can find them.
        pass

    prefer_system_installation = (
        os.getenv("RAPIDS_LIBCUFOREST_PREFER_SYSTEM_LIBRARY", "false").lower()
        != "false"
    )

    soname = "libcuforest++.so"
    libcuforest_lib = None
    if prefer_system_installation:
        # Prefer a system library if one is present to
        # avoid clobbering symbols that other packages might expect, but if no
        # other library is present use the one in the wheel.
        try:
            libcuforest_lib = _load_system_installation(soname)
        except OSError:
            libcuforest_lib = _load_wheel_installation(soname)
    else:
        # Prefer the libraries bundled in this package. If they aren't found
        # (which might be the case in builds where the library was prebuilt
        # before packaging the wheel), look for a system installation.
        try:
            libcuforest_lib = _load_wheel_installation(soname)
            if libcuforest_lib is None:
                libcuforest_lib = _load_system_installation(soname)
        except OSError:
            # If none of the searches above succeed, just silently return None
            # and rely on other mechanisms (like RPATHs on other DSOs) to
            # help the loader find the library.
            pass

    # The caller almost never needs to do anything with these libraries, but no
    # harm in offering the option since these objects at least provide handles
    # to inspect where libcuforest was loaded from.
    return libcuforest_lib
