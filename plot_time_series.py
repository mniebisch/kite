import json
import pathlib
from typing import Dict, List, Union

import click
import pandas as pd
import plotly.graph_objects as go


def extract_time_series_values(
    time_series_data: Dict[str, Union[int, List[float], float, Dict]]
) -> List[float]:
    x_max = time_series_data["maxX"]
    if not isinstance(x_max, int):
        raise ValueError
    if not isinstance(time_series_data["vertices"], list):
        raise ValueError
    values = time_series_data["vertices"][1 : x_max * 4][::4]
    return values


def extract_time_series_index(
    time_series_data: Dict[str, Union[int, List[float], float, Dict]]
) -> int:
    index = time_series_data["index"]
    if not isinstance(index, int):
        raise ValueError
    return index


def extract_time_series_time(
    time_series_data: Dict[str, Union[int, List[float], float, Dict]]
) -> List[float]:
    x_max = time_series_data["maxX"]
    if not isinstance(x_max, int):
        raise ValueError
    if not isinstance(time_series_data["vertices"], list):
        raise ValueError
    time_stamps = time_series_data["vertices"][0 : x_max * 4][::4]

    return time_stamps


def compose_time_series_data(
    time_series_data: Dict[str, Union[int, List[float], float, Dict]]
) -> pd.DataFrame:
    time_series_values = extract_time_series_values(time_series_data)
    time_series_index = extract_time_series_index(time_series_data)
    time_series_time = extract_time_series_time(time_series_data)

    time_series_df = pd.DataFrame(
        {
            "index": time_series_index,
            "time": time_series_time,
            "values": time_series_values,
        }
    )

    return time_series_df


def create_time_series_df(
    time_series_data: List[Dict[str, Union[int, List[float], float, Dict]]]
) -> pd.DataFrame:
    time_series_df = pd.concat(
        [compose_time_series_data(ts_data) for ts_data in time_series_data],
        ignore_index=True,
    )
    return time_series_df


def validate_input(input_data):
    if not isinstance(input_data, list):
        raise ValueError("Input should be a list.")

    for element in input_data:
        if not isinstance(element, dict):
            raise ValueError("Each element in the list should be a dictionary.")

        required_keys = {"index", "vertices", "minX", "maxX"}
        if not required_keys.issubset(element.keys()):
            raise ValueError(f"Each dictionary should have the keys: {required_keys}")


@click.command()
@click.option(
    "--file-path",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=False, file_okay=True
    ),
)
def main(file_path: pathlib.Path) -> None:
    with open(file_path, "r") as f:
        time_series_data = json.load(f)
        time_series_data = time_series_data["diagramLines"]

    validate_input(time_series_data)

    time_series_df = create_time_series_df(time_series_data)

    layout = go.Layout(
        hoversubplots="axis",
        hovermode="x unified",
        grid={"rows": len(time_series_df["index"].unique()), "columns": 1},
    )

    data = []
    for i in time_series_df["index"].unique():
        df = time_series_df[time_series_df["index"] == i]
        yaxis = "y" if i == 0 else f"y{i+1}"
        data.append(
            go.Scatter(
                x=df["time"],
                y=df["values"],
                xaxis="x",
                yaxis=yaxis,
                mode="lines",
                name=f"Index {i}",
            )
        )

    fig = go.Figure(data=data, layout=layout)

    fig.show()


if __name__ == "__main__":
    main()
