"""Dataset editing operations and cache invalidation."""

from pathlib import Path

import pandas as pd

from modules.large_data_engine import create_dataset_store
from services.session_service import DATASET_SCOPED_KEYS, DEFAULT_SESSION_STATE


ID_COLUMN_NAMES = {"index", "id", "customer id", "s.no", "sr.no", "row_id"}


def resolve_row_position(dataframe, row_input):
    """Resolve a row by a common ID column or its zero-based display position."""
    value = str(row_input).strip()
    if not value:
        return None

    id_columns = [
        column for column in dataframe.columns
        if str(column).lower() in ID_COLUMN_NAMES
    ]
    for column in id_columns:
        matches = dataframe.index[
            dataframe[column].astype(str).str.strip() == value
        ].tolist()
        if matches:
            return dataframe.index.get_loc(matches[0])

    try:
        position = int(value)
    except ValueError:
        return None
    if 0 <= position < len(dataframe):
        return position
    if 1 <= position <= len(dataframe):
        return position - 1
    return None


def _coerce_cell_value(series, value):
    if pd.api.types.is_bool_dtype(series.dtype):
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        if not normalized:
            return pd.NA
        raise ValueError("Enter true or false for this boolean column.")

    if pd.api.types.is_numeric_dtype(series.dtype):
        if not str(value).strip():
            return pd.NA
        try:
            number = pd.to_numeric(value, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("Enter a valid number for this numeric column.") from error
        if pd.api.types.is_integer_dtype(series.dtype) and float(number).is_integer():
            return int(number)
        return number

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        if not str(value).strip():
            return pd.NaT
        try:
            return pd.to_datetime(value, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("Enter a valid date or time for this column.") from error

    return value


def edit_cell(dataframe, row_input, column, value):
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' was not found.")
    position = resolve_row_position(dataframe, row_input)
    if position is None:
        raise ValueError(f"Row '{row_input}' was not found.")

    updated = dataframe.copy()
    converted = _coerce_cell_value(updated[column], value)
    if (
        pd.api.types.is_integer_dtype(updated[column].dtype)
        and isinstance(converted, float)
    ):
        updated[column] = updated[column].astype("float64")
    elif pd.isna(converted) and pd.api.types.is_integer_dtype(updated[column].dtype):
        updated[column] = updated[column].astype("Int64")
    updated.iat[position, updated.columns.get_loc(column)] = converted
    return updated


def add_record(dataframe, values):
    row = {}
    for column in dataframe.columns:
        value = values.get(column, "")
        if not str(value).strip() and pd.api.types.is_numeric_dtype(dataframe[column].dtype):
            row[column] = pd.NA
        elif not str(value).strip() and pd.api.types.is_datetime64_any_dtype(
            dataframe[column].dtype
        ):
            row[column] = pd.NaT
        else:
            row[column] = _coerce_cell_value(dataframe[column], value)
    new_row = pd.DataFrame([row], columns=dataframe.columns)
    updated = pd.concat([dataframe, new_row], ignore_index=True)
    for column in dataframe.columns:
        dtype = dataframe[column].dtype
        added_value = row[column]
        if pd.api.types.is_integer_dtype(dtype):
            if pd.isna(added_value):
                updated[column] = updated[column].astype("Int64")
            else:
                updated[column] = pd.to_numeric(updated[column], errors="raise")
        elif pd.api.types.is_numeric_dtype(dtype):
            updated[column] = pd.to_numeric(updated[column], errors="raise")
        elif pd.api.types.is_bool_dtype(dtype) and pd.isna(added_value):
            updated[column] = updated[column].astype("boolean")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            updated[column] = pd.to_datetime(updated[column], errors="raise")
    return updated


def delete_record(dataframe, row_input):
    position = resolve_row_position(dataframe, row_input)
    if position is None:
        raise ValueError(f"Row '{row_input}' was not found.")
    return dataframe.iloc[
        [index for index in range(len(dataframe)) if index != position]
    ].reset_index(drop=True)


def add_column(dataframe, column_name, default_value):
    column = str(column_name).strip()
    if not column:
        raise ValueError("Enter a column name.")
    if column in dataframe.columns:
        raise ValueError(f"Column '{column}' already exists.")
    updated = dataframe.copy()
    updated[column] = default_value
    return updated


def delete_column(dataframe, column):
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' was not found.")
    if len(dataframe.columns) <= 1:
        raise ValueError("The final column cannot be deleted.")
    return dataframe.drop(columns=[column])


def change_column_data_type(dataframe, column, data_type):
    """Convert one column to a supported type, rejecting values that cannot convert."""
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' was not found.")

    supported_types = {
        "Text",
        "Whole number",
        "Decimal number",
        "Date/time",
        "Boolean",
    }
    if data_type not in supported_types:
        raise ValueError("Choose a supported data type.")

    source = dataframe[column]
    non_empty = source.notna() & source.astype("string").str.strip().ne("")
    values = source.where(non_empty, pd.NA)

    if data_type == "Text":
        converted = source.astype("string")
    elif data_type in {"Whole number", "Decimal number"}:
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = non_empty & numeric.isna()
        if invalid.any():
            examples = source[invalid].astype(str).drop_duplicates().head(3).tolist()
            raise ValueError(
                f"Cannot convert '{column}' to {data_type.lower()}; "
                f"invalid value(s): {', '.join(examples)}."
            )
        if data_type == "Whole number":
            fractional = numeric.notna() & numeric.mod(1).ne(0)
            if fractional.any():
                examples = (
                    source[fractional]
                    .astype(str)
                    .drop_duplicates()
                    .head(3)
                    .tolist()
                )
                raise ValueError(
                    f"Cannot convert '{column}' to whole numbers; "
                    f"non-whole value(s): {', '.join(examples)}."
                )
            converted = numeric.astype("Int64")
        else:
            converted = numeric.astype("float64")
    elif data_type == "Date/time":
        dates = pd.to_datetime(values, errors="coerce")
        invalid = non_empty & dates.isna()
        if invalid.any():
            examples = source[invalid].astype(str).drop_duplicates().head(3).tolist()
            raise ValueError(
                f"Cannot convert '{column}' to date/time; "
                f"invalid value(s): {', '.join(examples)}."
            )
        converted = dates
    else:
        normalized = source.astype("string").str.strip().str.lower()
        boolean_values = {
            "true": True, "1": True, "yes": True, "y": True,
            "false": False, "0": False, "no": False, "n": False,
        }
        converted = normalized.map(boolean_values).astype("boolean")
        invalid = non_empty & converted.isna()
        if invalid.any():
            examples = source[invalid].astype(str).drop_duplicates().head(3).tolist()
            raise ValueError(
                f"Cannot convert '{column}' to boolean; use true/false, "
                f"yes/no, or 1/0. Invalid value(s): {', '.join(examples)}."
            )

    updated = dataframe.copy()
    updated[column] = converted
    return updated


def merge_tables_on_common_column(dataframe, tables, column, how="outer"):
    """Sequentially merge additional tables while preserving duplicate column names."""
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' was not found in the current dataset.")
    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError("Choose a supported merge type.")
    if not dataframe.columns.is_unique:
        raise ValueError("The current dataset has duplicate column names and cannot be merged safely.")

    merged = dataframe.copy()
    for table_name, table in tables:
        if column not in table.columns:
            raise ValueError(
                f"Column '{column}' is missing from '{Path(table_name).name}'."
            )
        if not table.columns.is_unique:
            raise ValueError(
                f"'{Path(table_name).name}' has duplicate column names and cannot be merged safely."
            )

        overlapping_columns = (
            set(merged.columns).intersection(table.columns) - {column}
        )
        if overlapping_columns:
            suffix = "".join(
                character if character.isalnum() else "_"
                for character in Path(table_name).stem
            ).strip("_") or "table"
            right_columns = table.columns.tolist()
            reserved_names = set(merged.columns).union(right_columns)
            rename = {}
            for name in right_columns:
                if name not in overlapping_columns:
                    continue
                candidate = f"{name}_{suffix}"
                duplicate_index = 2
                while candidate in reserved_names:
                    candidate = f"{name}_{suffix}_{duplicate_index}"
                    duplicate_index += 1
                rename[name] = candidate
                reserved_names.add(candidate)
            table = table.rename(columns=rename)

        merged = merged.merge(table, on=column, how=how, sort=False)
    return merged


def _invalidate_analysis(state):
    preserved = {
        "last_file_id",
        "original_dataframe",
        "working_dataframe",
        "dataset_store",
        "upload_merge_signature",
        "upload_merge_base_dataframe",
    }
    for key in DATASET_SCOPED_KEYS:
        if key not in preserved:
            state[key] = DEFAULT_SESSION_STATE[key]
    state["df_cleaned"] = state["working_dataframe"].copy()
    state["cleaning_applied"] = False


def update_working_dataset(state, dataframe, source_name, sheet_name=None):
    """Persist edits and rebuild all dataset-derived analysis state."""
    fingerprint_bytes = (
        dataframe.to_csv(index=False, lineterminator="\n").encode("utf-8")
        + repr([(str(column), str(dtype)) for column, dtype in dataframe.dtypes.items()]).encode(
            "utf-8"
        )
    )
    dataset_store = create_dataset_store(
        dataframe=dataframe,
        file_bytes=fingerprint_bytes,
        source_name=source_name,
        sheet_name=sheet_name,
    )
    state["working_dataframe"] = dataframe
    state["dataset_store"] = dataset_store
    _invalidate_analysis(state)


def restore_uploaded_dataset(state, file_bytes, source_name, sheet_name=None):
    """Restore the selected uploaded file and rebuild its original dataset store."""
    state["upload_merge_signature"] = None
    state["upload_merge_base_dataframe"] = None
    dataframe = state["original_dataframe"].copy()
    dataset_store = create_dataset_store(
        dataframe=dataframe,
        file_bytes=file_bytes,
        source_name=source_name,
        sheet_name=sheet_name,
    )
    state["working_dataframe"] = dataframe
    state["dataset_store"] = dataset_store
    _invalidate_analysis(state)
