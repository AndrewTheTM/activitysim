# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

import itertools
import logging
import os
from typing import Literal

import pandas as pd

from activitysim.core import (
    config,
    estimation,
    expressions,
    logit,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings
from activitysim.core.configuration.logit import LogitComponentSettings
from activitysim.core.interaction_simulate import interaction_simulate

logger = logging.getLogger(__name__)

def annotate_vehicle_type_choice_households(
    state: workflow.State, model_settings: VehicleTypeChoiceSettings, trace_label: str
):
    """
    Add columns to the households table in the pipeline according to spec.

    Parameters
    ----------
    state : workflow.State
    model_settings : VehicleTypeChoiceSettings
    trace_label : str
    """
    households = state.get_dataframe("households")
    expressions.assign_columns(
        state,
        df=households,
        model_settings=model_settings.annotate_households,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_households"),
    )
    state.add_table("households", households)


def annotate_vehicle_type_choice_persons(
    state: workflow.State, model_settings: VehicleTypeChoiceSettings, trace_label: str
):
    """
    Add columns to the persons table in the pipeline according to spec.

    Parameters
    ----------
    state : workflow.State
    model_settings : VehicleTypeChoiceSettings
    trace_label : str
    """
    persons = state.get_dataframe("persons")
    expressions.assign_columns(
        state,
        df=persons,
        model_settings=model_settings.annotate_persons,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
    )
    state.add_table("persons", persons)


def annotate_vehicle_type_choice_vehicles(
    state: workflow.State, model_settings: VehicleTypeChoiceSettings, trace_label: str
):
    """
    Add columns to the vehicles table in the pipeline according to spec.

    Parameters
    ----------
    state : workflow.State
    model_settings : VehicleTypeChoiceSettings
    trace_label : str
    """
    vehicles = state.get_dataframe("vehicles")
    expressions.assign_columns(
        state,
        df=vehicles,
        model_settings=model_settings.annotate_vehicles,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_vehicles"),
    )
    state.add_table("vehicles", vehicles)

class VehicleTypeChoiceSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `vehicle_type_choice` component.
    """

    VEHICLE_TYPE_DATA_FILE: str | None = None
    PROBS_SPEC: str | None = None
    combinatorial_alts: dict | None = None
    preprocessor: PreprocessorSettings | None = None
    alts_preprocessor: PreprocessorSettings | None = None
    SIMULATION_TYPE: Literal[
        "simple_simulate", "interaction_simulate"
    ] = "interaction_simulate"
    COLS_TO_INCLUDE_IN_VEHICLE_TABLE: list[str] = []

    COLS_TO_INCLUDE_IN_CHOOSER_TABLE: list[str] = []
    """Columns to include in the chooser table for use in utility calculations."""
    COLS_TO_INCLUDE_IN_ALTS_TABLE: list[str] = []
    """Columns to include in the alternatives table for use in utility calculations."""

    annotate_households: PreprocessorSettings | None = None
    annotate_persons: PreprocessorSettings | None = None
    annotate_vehicles: PreprocessorSettings | None = None

    REQUIRE_DATA_FOR_ALL_ALTS: bool = False
    WRITE_OUT_ALTS_FILE: bool = False

    FLEET_YEAR: int

    explicit_chunk: float = 0
    """
    If > 0, use this chunk size instead of adaptive chunking.
    If less than 1, use this fraction of the total number of rows.
    """

@workflow.step
def transcad_vehicle_type_choice(
    state: workflow.State,
    persons: pd.DataFrame,
    households: pd.DataFrame,
    vehicles: pd.DataFrame,
    vehicles_merged: pd.DataFrame,
    model_settings: VehicleTypeChoiceSettings | None = None,
    model_settings_file_name: str = "vehicle_type_choice.yaml",
    trace_label: str = "vehicle_type_choice",
) -> None:
    """Assign a vehicle type to each vehicle in the `vehicles` table.

    If a "SIMULATION_TYPE" is set to simple_simulate in the
    vehicle_type_choice.yaml config file, then the model specification .csv file
    should contain one column of coefficients for each distinct alternative. This
    format corresponds to ActivitySim's :func:`activitysim.core.simulate.simple_simulate`
    format. Otherwise, this model will construct a table of alternatives, at run time,
    based on all possible combinations of values of the categorical variables enumerated
    as "combinatorial_alts" in the .yaml config. In this case, the model leverages
    ActivitySim's :func:`activitysim.core.interaction_simulate` model design, in which
    the model specification .csv has only one column of coefficients, and the utility
    expressions can turn coefficients on or off based on attributes of either
    the chooser _or_ the alternative.

    As an optional second step, the user may also specify a "PROBS_SPEC" .csv file in
    the main .yaml config, corresponding to a lookup table of additional vehicle
    attributes and probabilities to be sampled and assigned to vehicles after the logit
    choices have been made. The rows of the "PROBS_SPEC" file must include all body type
    and vehicle age choices assigned in the logit model. These additional attributes are
    concatenated with the selected alternative from the logit model to form a single
    vehicle type name to be stored in the `vehicles` table as the vehicle_type column.

    Only one household vehicle is selected at a time to allow for the introduction of
    owned vehicle related attributes. For example, a household may be less likely to
    own a second van if they already own one. The model is run sequentially through
    household vehicle numbers. The preprocessor is run for each iteration on the entire
    vehicles table to allow for computation of terms involving the presence of other
    household vehicles.

    The user may also augment the `households` or `persons` tables with new vehicle
    type-based fields specified via expressions in "annotate_households_vehicle_type.csv"
    and "annotate_persons_vehicle_type.csv", respectively.

    Parameters
    ----------
    state : workflow.State
    persons : pd.DataFrame
    households : pd.DataFrame
    vehicles : pd.DataFrame
    vehicles_merged :pd. DataFrame
    model_settings : class specifying the model settings
    model_settings_file_name: filename of the model settings file
    trace_label: trace label of the vehicle type choice model
    """
    if model_settings is None:
        model_settings = VehicleTypeChoiceSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    estimator = estimation.manager.begin_estimation(state, "vehicle_type")

    model_spec_raw = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec_raw, coefficients_df, estimator
    )

    constants = config.get_model_constants(model_settings)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(coefficients_df)

    # choices, choosers = iterate_vehicle_type_choice(
    #     state,
    #     vehicles_merged,
    #     model_settings,
    #     model_spec,
    #     locals_dict,
    #     estimator,
    #     state.settings.chunk_size,
    #     trace_label,
    # )

    

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

        # FIXME #interaction_simulate_estimation_requires_chooser_id_in_df_column
        #  shuold we do it here or have interaction_simulate do it?
        # chooser index must be duplicated in column or it will be omitted from interaction_dataset
        # estimation requires that chooser_id is either in index or a column of interaction_dataset
        # so it can be reformatted (melted) and indexed by chooser_id and alt_id
        assert choosers.index.name == "vehicle_id"
        assert "vehicle_id" not in choosers.columns
        choosers["vehicle_id"] = choosers.index

        # FIXME set_alt_id - do we need this for interaction_simulate estimation bundle tables?
        estimator.set_alt_id("alt_id")
        estimator.set_chooser_id(choosers.index.name)

        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "vehicles", "vehicle_type_choice"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # update vehicles table
    vehicles = pd.concat([vehicles, choices], axis=1)
    state.add_table("vehicles", vehicles)

    # - annotate tables
    if model_settings.annotate_households:
        annotate_vehicle_type_choice_households(state, model_settings, trace_label)
    if model_settings.annotate_persons:
        annotate_vehicle_type_choice_persons(state, model_settings, trace_label)
    if model_settings.annotate_vehicles:
        annotate_vehicle_type_choice_vehicles(state, model_settings, trace_label)

    tracing.print_summary(
        "vehicle_type_choice", vehicles.vehicle_type, value_counts=True
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(
            vehicles, label="vehicle_type_choice", warn_if_empty=True
        )
