#!/usr/bin/env python3
"""Convert Swiss TLM road or railway lines into GeoParquet with line width metadata."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pandas as pd

from utilities import define_bridge_parameters

LOGGER = logging.getLogger(__name__)

WIDTH_RE = re.compile(r"(\d+(?:[\.,]\d+)?)")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def infer_source_layer(path: str) -> str:
    name = Path(path).name
    if name.endswith(".zip"):
        name = name[:-4]
    if name.endswith(".gpkg"):
        name = name[:-5]
    return name


def coerce_width(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        match = WIDTH_RE.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def to_optional_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.strip()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def determine_width(length_m: float, anzahl_spuren: Optional[int], objektart: Optional[str]) -> Optional[float]:
    try:
        deck_width, *_ = define_bridge_parameters(
            length_m,
            anzahl_spuren=anzahl_spuren,
            objektart=objektart,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Width fallback for objektart=%s: %s", objektart, exc)
        deck_width = None
    width = coerce_width(deck_width)
    if width is None and objektart:
        width = coerce_width(objektart)
    return width


def resolve_path(input_path: str) -> str:
    path = Path(input_path)
    if path.suffix == ".zip":
        return f"/vsizip/{path.resolve()}"
    return str(path)


def process_dataset(input_path: str, output_path: str) -> None:
    LOGGER.info("Reading %s", input_path)
    vsi_path = resolve_path(input_path)
    gdf = gpd.read_file(vsi_path)

    if gdf.empty:
        LOGGER.warning("No features found in %s", input_path)
        gdf.to_parquet(
            output_path,
            index=False,
            schema_version="1.1.0",
            write_covering_bbox=True,
            compression="snappy",
        )
        return

    gdf = gdf[~gdf.geometry.isna()].copy()

    if "anzahl_spuren" in gdf.columns:
        gdf["anzahl_spuren"] = pd.to_numeric(gdf["anzahl_spuren"], errors="coerce").astype("Int64")
    else:
        gdf["anzahl_spuren"] = pd.Series(pd.NA, index=gdf.index, dtype="Int64")

    if "objektart" not in gdf.columns:
        gdf["objektart"] = pd.Series([None] * len(gdf), index=gdf.index)

    length_2d = gdf.geometry.length
    gdf["length_2d_m"] = length_2d

    gdf["line_width_m"] = [
        determine_width(length, to_optional_int(spuren), objektart)
        for length, spuren, objektart in zip(
            length_2d,
            gdf["anzahl_spuren"],
            gdf["objektart"],
        )
    ]

    gdf["geometry_blob"] = gdf.geometry.apply(lambda geom: geom.wkb if geom is not None else None)
    gdf["source_layer"] = infer_source_layer(input_path)
    gdf["crs_epsg"] = gdf.crs.to_epsg() if gdf.crs else None

    LOGGER.info("Writing %s features to %s", len(gdf), output_path)
    gdf.to_parquet(
        output_path,
        index=False,
        schema_version="1.1.0",
        write_covering_bbox=True,
        compression="snappy",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GeoParquet for Swiss roads or railways")
    parser.add_argument("input_path", help="Path to roads.gpkg.zip or railways.gpkg.zip")
    parser.add_argument("output_path", help="Destination Parquet file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    output_path = args.output_path
    if not output_path.lower().endswith(".parquet"):
        output_path = f"{output_path}.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    process_dataset(args.input_path, output_path)


if __name__ == "__main__":
    main()
