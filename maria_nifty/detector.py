"""Detector geometry, masking, and grouping helpers.

The classes and functions in this module build detector layouts from hex-ring
patterns or from measured TOD data, apply detector masks, and construct
detector groups used downstream by the response models and plots.
"""

import numpy as np
from maria.tod import TOD
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

S75 = np.sqrt(0.75)

COORD_UPDATE = np.array([[S75, -0.5], [0, -1], [-S75, -0.5], [-S75, 0.5], [0, 1], [S75, 0.5]])


def n_coords(order):
    """Return the number of detector coordinates in a hex ring of `order`."""
    return 1 + 6 * sum(range(1, order + 1))


def hex_ring_clockwise(order, center=(0, 0), distances=None):
    """Build hex-ring detector coordinates ordered clockwise.

    Parameters
    ----------
    order : int
        Hex-ring order; 0 returns a single central coordinate.
    center : tuple of float, optional
        (x, y) origin of the layout.
    distances : float or None, optional
        Pixel spacing; defaults to ``1 / order`` when order > 0.

    Returns
    -------
    numpy.ndarray, shape (n_coords(order), 2)
        Detector coordinates in clockwise traversal order.
    """
    coords = np.zeros((n_coords(order), 2))

    for o in range(order + 1):
        o0 = n_coords(o - 1) if o > 0 else 0
        coords[o0] = np.array(center) + np.array([0, o])

        for i in range(1, 6 * o):
            upd_idx = (i - 1) // o
            coords[o0 + i] = coords[o0 + i - 1] + COORD_UPDATE[upd_idx]

    if order > 0:
        distances = 1 / order if distances is None else distances
        coords = coords * distances

    return coords


def hex_ring_linear(order, center=(0, 0), distances=None):
    """Build hex-ring detector coordinates in row-major (linear) order.

    Parameters
    ----------
    order : int
        Hex-ring order; 0 returns a single central coordinate.
    center : tuple of float, optional
        (x, y) origin of the layout.
    distances : float or None, optional
        Pixel spacing; defaults to ``1 / order`` when order > 0.

    Returns
    -------
    numpy.ndarray, shape (n_coords(order), 2)
        Detector coordinates in row-major traversal order.
    """
    coords = np.zeros((n_coords(order), 2))

    for o in range(-order, order + 1):
        o0 = o0 + 2 * order + 1 - np.abs(o - 1) if o > -order else 0  # noqa: F821
        c0 = c0 + COORD_UPDATE[1] if o > 0 else c0 + COORD_UPDATE[0] if o > -order else np.array([0, order])  # noqa: F821
        coords[o0] = np.array(center) + c0

        for i in range(1, 1 + 2 * order - np.abs(o)):
            coords[o0 + i] = c0 + i * COORD_UPDATE[2]

    if order > 0:
        distances = 1 / order if distances is None else distances
        coords = coords * distances

    return coords


def hex_ring_mapping(order, linear2ring=True):
    """Return index mappings between linear and clockwise hex-ring orderings.

    Parameters
    ----------
    order : int
        Hex-ring order of the layout.
    linear2ring : bool, optional
        If ``True``, return indices that convert linear → clockwise order.
        If ``False``, return the inverse (clockwise → linear).

    Returns
    -------
    numpy.ndarray of int, shape (n_coords(order),)
        Index array such that ``clockwise[i] = linear[indices[i]]`` (or the
        reverse when ``linear2ring=False``).
    """
    top_coords = hex_ring_linear(order)
    ring_coords = hex_ring_clockwise(order)

    if linear2ring:
        tree = KDTree(top_coords)
        _, indices = tree.query(ring_coords)
    else:
        tree = KDTree(ring_coords)
        _, indices = tree.query(top_coords)

    return indices


def hex_coordinates(order, mode="linear", center=(0, 0), distances=None):
    """Return detector coordinates for the requested hex-ring layout.

    Parameters
    ----------
    order : int
        Hex-ring order of the layout.
    mode : {"linear", "clockwise"}, optional
        Traversal order of the output coordinates.
    center : tuple of float, optional
        (x, y) origin of the layout.
    distances : float or None, optional
        Pixel spacing; defaults to ``1 / order`` when order > 0.

    Returns
    -------
    numpy.ndarray, shape (n_coords(order), 2)
        Detector coordinates in the requested ordering.
    """
    if mode == "linear":
        coordinates = hex_ring_linear(order, center, distances)
    elif mode == "clockwise":
        coordinates = hex_ring_clockwise(order, center, distances)
    else:
        raise ValueError(f"Invalid mode: `{mode}`")
    return coordinates


def detector_mask(size: int, mask=None, mask_indices=None) -> np.ndarray:
    """Normalize detector mask input into a boolean mask array.

    Parameters
    ----------
    size : int
        Total number of detectors (length of the returned array).
    mask : array-like of bool or None, optional
        Pre-built boolean mask aligned with the detector axis.
    mask_indices : array-like of int or None, optional
        Indices of detectors to mask (``True`` in the output).

    Returns
    -------
    numpy.ndarray of bool, shape (size,)
        Boolean mask where ``True`` marks a masked (excluded) detector.
    """
    if mask is not None and mask_indices is not None:
        raise ValueError("Use either `mask` or `mask_indices`, not both.")

    if mask_indices is not None:
        idx = np.asarray(mask_indices, dtype=int)
        out = np.zeros(size, dtype=bool)
        if idx.size == 0:
            return out
        if np.any(idx < 0) or np.any(idx >= size):
            raise ValueError("`mask_indices` contains out-of-range indices.")
        out[idx] = True
        return out

    if mask is None:
        return np.zeros(size, dtype=bool)

    out = np.asarray(mask, dtype=bool)
    if out.shape != (size,):
        raise ValueError("Expected `mask` to match detector axis shape `(n_detectors,)`.")
    return out


class DetectorGroups:
    """Detector grouping metadata with normalized group indices and masks."""

    def __init__(self, indices, order=None, centers=None, mask=None):
        indices = np.asarray(indices, dtype=int).copy()
        mask = detector_mask(indices.shape[0], mask=mask)

        indices[mask] = -1

        # Normalize unmasked labels to a contiguous zero-based range (0..n_groups-1)
        # while preserving -1 for masked coordinates.
        valid = indices >= 0
        normalized = np.full(indices.shape, -1, dtype=int)
        if np.any(valid):
            raw_values = np.unique(indices[valid])
            normalized[valid] = np.searchsorted(raw_values, indices[valid])
            self.values, self.sizes = np.unique(normalized[valid], return_counts=True)
        else:
            self.values = np.array([], dtype=int)
            self.sizes = np.array([], dtype=int)

        self.indices = normalized
        self.number = self.values.size
        self.mask = mask
        self.n_masked = int(np.count_nonzero(mask))

        self.order = order
        self.centers = centers

    @classmethod
    def build(cls, coordinates, indices, tods=None, from_tods: bool = False, **kwargs) -> "DetectorGroups":
        """Build detector groups from geometry or TOD clustering.

        Parameters
        ----------
        coordinates : array-like
            Detector coordinates used for hex-based grouping.
        indices : array-like
            Detector ordering indices used when grouping from TODs.
        tods : maria.tod.TOD or numpy.ndarray, optional
            TOD input used when grouping detectors by similarity.
        from_tods : bool, optional
            If `True`, build groups from TOD clustering instead of geometry.
        **kwargs
            Additional keyword arguments forwarded to the chosen constructor.

        Returns
        -------
        DetectorGroups
            Detector grouping metadata.
        """
        if from_tods:
            return cls.from_tods(indices, tods, **kwargs)
        else:
            return cls.from_hex(coordinates, **kwargs)

    @classmethod
    def from_hex(
        cls, coordinates, *, order: int = -1, mode: str = "linear", center=(0, 0), distances=None, mask=None
    ) -> "DetectorGroups":
        """Group detectors by nearest hex-ring center coordinates.

        Parameters
        ----------
        coordinates : array-like
            Detector coordinates to assign to hex-ring centers.
        order : int, optional
            Hex-ring order used to generate the reference layout. Use ``-1``
            to treat each detector as its own group center.
        mode : str, optional
            Ordering mode for the reference layout.
        center : tuple of float, optional
            Center of the reference layout.
        distances : float or tuple of float, optional
            Pixel spacing used for the reference layout.
        mask : array-like, optional
            Boolean detector mask applied to the resulting groups.

        Returns
        -------
        DetectorGroups
            Detector groups derived from the reference hex layout.
        """
        centers = np.asarray(coordinates) if order == -1 else hex_coordinates(order, mode, center, distances)

        tree = KDTree(centers)
        _, group_indices = tree.query(coordinates)

        return cls(group_indices, order, centers, mask)

    @classmethod
    def from_tods(
        cls,
        indices,
        tods,
        *,
        reorder: bool = True,
        n_groups: int = 10,
        metric: str = "euclidean",
        method: str = "average",
        criterion: str = "maxclust",
        mask=None,
    ) -> "DetectorGroups":
        """Cluster detectors by TOD similarity and return normalized groups.

        Parameters
        ----------
        indices : array-like or None
            Optional detector reordering indices.
        tods : maria.tod.TOD or numpy.ndarray
            TOD data used for hierarchical clustering.
        reorder : bool, optional
            Whether to reorder TODs according to `indices` before clustering.
        n_groups : int, optional
            Requested number of detector groups.
        metric : str, optional
            Distance metric passed to :func:`scipy.spatial.distance.pdist`.
        method : str, optional
            Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`.
        criterion : str, optional
            Clustering criterion passed to :func:`scipy.cluster.hierarchy.fcluster`.
        mask : array-like, optional
            Boolean detector mask applied to the resulting groups.

        Returns
        -------
        DetectorGroups
            Detector groups derived from TOD clustering.
        """
        tod_data = tods
        if isinstance(tods, TOD):
            tod_data = np.array(tods.data["data"], dtype="float64")
        if not isinstance(tod_data, np.ndarray):
            raise ValueError("Expected `tods` to be an instance of `TOD` or a numpy array.")
        if reorder and indices is not None:
            tod_data = tod_data[indices]

        n_det = tod_data.shape[0]
        mask = detector_mask(n_det, mask=mask)

        valid = ~mask
        group_indices = np.full(n_det, -1, dtype=int)
        n_valid = int(np.count_nonzero(valid))

        # Cluster only unmasked detectors. Masked detectors keep index -1.
        if n_valid == 1:
            group_indices[valid] = 0
        elif n_valid > 1:
            dist = pdist(tod_data[valid], metric)
            Z = linkage(dist, method)
            clustered = fcluster(Z, t=n_groups, criterion=criterion)
            group_indices[valid] = clustered

        return cls(group_indices, None, None, mask)


class Detector:
    """Detector geometry plus mask and group metadata."""

    def __init__(self, coordinates, order=None, indices=None, mask=None, tods=None, groups=None):
        if groups is None:
            groups = dict()
        self.coordinates = np.asarray(coordinates)
        self.order = order
        self.indices = indices
        self.mask = detector_mask(self.coordinates.shape[0], mask=mask)
        self.add_groups(tods, **groups)

    @property
    def size(self):
        """Return the number of unmasked detectors."""
        return int(self.full_size - self.n_masked)

    @property
    def full_size(self):
        """Return the total number of detectors before masking."""
        return self.coordinates.shape[0]

    @property
    def n_masked(self):
        """Return the number of masked detectors."""
        return int(np.count_nonzero(self.mask))

    @property
    def n_groups(self):
        """Return the number of detector groups."""
        return self.groups.number

    @classmethod
    def build(cls, *, tods=None, from_tods: bool = False, **kwargs) -> "Detector":
        """Build a detector either from geometry or from TOD metadata.

        Parameters
        ----------
        tods : maria.tod.TOD, optional
            TOD object used when building from observed data.
        from_tods : bool, optional
            If `True`, derive coordinates and grouping from the TOD object.
        **kwargs
            Additional keyword arguments forwarded to the chosen constructor.

        Returns
        -------
        Detector
            A detector instance with coordinates, mask, and groups.
        """
        if from_tods:
            return cls.from_tods(tods=tods, **kwargs)
        else:
            return cls.from_hex(**kwargs)

    @classmethod
    def from_hex(
        cls,
        *,
        order: int,
        groups: dict = None,
        mode: str = "linear",
        center=(0, 0),
        distances=None,
        mask=None,
        mask_indices=None,
    ) -> "Detector":
        """Construct a detector from hex-ring geometry and optional masks.

        Parameters
        ----------
        order : int
            Hex-ring order of the detector layout.
        groups : dict, optional
            Grouping configuration forwarded to :class:`DetectorGroups`.
        mode : str, optional
            Hex-ring ordering mode.
        center : tuple of float, optional
            Detector layout center.
        distances : float or tuple of float, optional
            Pixel spacing of the detector layout.
        mask : array-like, optional
            Boolean detector mask.
        mask_indices : array-like, optional
            Indices of detectors to mask.

        Returns
        -------
        Detector
            A detector built from the requested hex layout.
        """
        if groups is None:
            groups = dict()
        coordinates = hex_coordinates(order, mode, center, distances)
        det_mask = detector_mask(coordinates.shape[0], mask=mask, mask_indices=mask_indices)
        return cls(coordinates, order, None, det_mask, None, groups)

    @classmethod
    def from_tods(
        cls,
        *,
        tods,
        groups: dict = None,
        max_order: int = 0,
        mode: str = "linear",
        center=(0, 0),
        distances=None,
        mask=None,
        mask_indices=None,
    ) -> "Detector":
        """Construct a detector from TOD coordinates and optional clustering.

        Parameters
        ----------
        tods : maria.tod.TOD
            TOD object that provides detector coordinates.
        groups : dict, optional
            Grouping configuration forwarded to :class:`DetectorGroups`.
        max_order : int, optional
            Maximum hex-ring order to try when matching the detector layout.
        mode : str, optional
            Hex-ring ordering mode used for geometry matching.
        center : tuple of float, optional
            Approximate detector layout center.
        distances : float or tuple of float, optional
            Pixel spacing of the detector layout.
        mask : array-like, optional
            Boolean detector mask.
        mask_indices : array-like, optional
            Indices of detectors to mask.

        Returns
        -------
        Detector
            A detector reconstructed from TOD coordinates.
        """
        if groups is None:
            groups = dict()
        if not isinstance(tods, TOD):
            raise ValueError("Expected `tods` to be an instance of `TOD`")

        coords = tods.dets.offsets
        input_size = coords.shape[0]
        input_mask = detector_mask(input_size, mask=mask, mask_indices=mask_indices)
        order = None
        indices = None

        if max_order > 0:
            # shift center coordinates to (0,0)
            tree = KDTree(coords)
            dist, idx = tree.query(center)
            center = coords[idx]
            shifted_coords = coords - center

            # unit distance between nearby coordinates
            tree = KDTree(shifted_coords)
            dist, _ = tree.query(shifted_coords, k=2)
            spacing = np.median(dist[:, 1])
            unit_coords = shifted_coords / spacing

            # rotate
            tree = KDTree(unit_coords)
            dist, idx = tree.query((0, 0.5), k=2)
            p1, p2 = unit_coords[idx]
            dx, dy = p2 - p1
            angle = np.arctan2(dy, dx)
            phi = -angle + np.pi / 2
            R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            rotated_coords = unit_coords @ R.T

            # get full detector
            order = None
            for o in range(max_order):
                full_detector = cls.build(order=o, mode=mode, distances=1)
                full_coords = full_detector.coordinates
                tree = KDTree(full_coords)
                dist, _ = tree.query(rotated_coords)
                if dist.max() < 0.1:
                    order = o
                    break

            if order is None:
                raise ValueError("Increase `max_order` argument.")

            # map rotated coordinates to the order of the full detector
            tree = KDTree(rotated_coords)
            dist, idx = tree.query(full_coords)
            overlap_mask = dist < 0.1
            indices = idx[overlap_mask]
            coords = rotated_coords[indices]

            if order > 0:
                distances = 1 / order if distances is None else distances
                coords = coords * distances

        final_mask = input_mask if indices is None else input_mask[indices]

        return cls(coords, order, indices, final_mask, tods, groups)

    def add_groups(self, tods=None, from_tods=False, **kwargs):
        """Update detector grouping metadata."""
        kwargs.setdefault("mask", self.mask)
        self.groups = DetectorGroups.build(self.coordinates, self.indices, tods, from_tods, **kwargs)
