import itertools
import json
import pathlib
from typing import Any, Dict, List, Tuple, TypeAlias, Union

import click
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from numpy import typing as npt

TimeSeriesData: TypeAlias = Dict[str, Union[int, List[float], float, Dict]]
TimeSeriesCollection: TypeAlias = List[TimeSeriesData]
KiteData: TypeAlias = List[Dict[str, Union[str, TimeSeriesCollection, Dict, int]]]


def extract_time_series_values(time_series_data: TimeSeriesData) -> List[float]:
    if not isinstance(time_series_data["vertices"], list):
        raise ValueError
    time_stamps = extract_time_series_time(time_series_data)
    x_max = time_stamps[-1]
    x_max_index = time_stamps.index(x_max)
    values = time_series_data["vertices"][1::4]
    values = values[: x_max_index + 1]

    return values


def extract_time_series_index(time_series_data: TimeSeriesData) -> int:
    index = time_series_data["index"]
    if not isinstance(index, int):
        raise ValueError
    return index


def extract_time_series_time(time_series_data: TimeSeriesData) -> List[float]:
    if not isinstance(time_series_data["vertices"], list):
        raise ValueError
    time_stamps = time_series_data["vertices"][::4]
    x_max = max(time_stamps)
    x_max_index = time_stamps.index(x_max)
    time_stamps = time_stamps[: x_max_index + 1]

    return time_stamps


def compose_time_series_data(time_series_data: TimeSeriesData) -> pd.DataFrame:
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


def create_time_series_df(time_series_data: TimeSeriesCollection) -> pd.DataFrame:
    time_series_df = pd.concat(
        [compose_time_series_data(ts_data) for ts_data in time_series_data],
        ignore_index=True,
    )
    return time_series_df


def validate_input(input_data: TimeSeriesCollection) -> None:
    if not isinstance(input_data, list):
        raise ValueError("Input should be a list.")

    for element in input_data:
        if not isinstance(element, dict):
            raise ValueError("Each element in the list should be a dictionary.")

        required_keys = {"index", "vertices", "minX", "maxX"}
        if not required_keys.issubset(element.keys()):
            raise ValueError(f"Each dictionary should have the keys: {required_keys}")


def process_kite_data(kite_data: KiteData) -> pd.DataFrame:
    observations = []
    for observation_index, observations_data in enumerate(kite_data):
        time_series_data = observations_data["diagramLines"]
        if not isinstance(time_series_data, list):
            raise ValueError("Each element in the list should be a dictionary")

        validate_input(time_series_data)
        observation = create_time_series_df(time_series_data)
        observation["observation"] = observation_index
        observations.append(observation)

    return pd.concat(observations, ignore_index=True)


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

    time_series_df = process_kite_data(time_series_data)

    variable_names = np.sort(time_series_df["index"].unique())
    observations = np.sort(time_series_df["observation"].unique())

    app = Dash(__name__)
    app.layout = html.Div(
        [
            "Observations:",
            dcc.Dropdown(
                observations,
                observations[0],
                id="time_series_observations",
            ),
            "Variables:",
            dcc.Dropdown(
                variable_names,
                variable_names,
                id="time_series_variables",
                multi=True,
            ),
            dcc.Graph(
                id="time_series_figure",
                style={"height": "85vh", "width": "90vw"},
            ),
        ],
    )

    @app.callback(
        Output("time_series_figure", "figure"),
        Input("time_series_variables", "value"),
    )
    def update_time_series_figure(selected_variables):
        selected_variables = sorted(selected_variables)
        color_map = {
            var_name: color
            for var_name, color in zip(
                variable_names, itertools.cycle(plotly.colors.qualitative.Plotly)
            )
        }
        layout = go.Layout(
            hoversubplots="axis",
            hovermode="x unified",
            grid={"rows": len(selected_variables), "columns": 1},
        )

        data = []
        for i, var_name in enumerate(selected_variables):
            df = time_series_df[time_series_df["index"] == var_name]
            yaxis = "y" if i == 0 else f"y{i+1}"
            data.append(
                go.Scatter(
                    x=df["time"],
                    y=df["values"],
                    xaxis="x",
                    yaxis=yaxis,
                    mode="lines",
                    name=f"Index {var_name}",
                    line=dict(color=color_map[var_name]),
                )
            )

        fig = go.Figure(data=data, layout=layout)
        return fig

    @app.callback(
        Output("time_series_variables", "options"),
        Output("time_series_variables", "value"),
        Input("time_series_observations", "value"),
    )
    def update_time_series_variables(
        selected_observation: Any,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        selected_variables = time_series_df[
            time_series_df["observation"] == selected_observation
        ]["index"].unique()
        selected_variables = np.sort(selected_variables)
        return selected_variables, selected_variables

    app.run(debug=True)


if __name__ == "__main__":
    main()
