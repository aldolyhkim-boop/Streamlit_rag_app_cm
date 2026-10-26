"""Manual data editing controls for the active uploaded dataset."""

import pandas as pd
import streamlit as st

from services.data_manipulation_service import (
    add_column,
    add_record,
    change_column_data_type,
    delete_column,
    delete_record,
    edit_cell,
    merge_tables_on_common_column,
    restore_uploaded_dataset,
    update_working_dataset,
)
from services.file_service import (
    SUPPORTED_EXTENSIONS,
    get_excel_sheet_names,
    load_dataframe,
    validate_uploaded_file,
)


@st.cache_data(max_entries=8)
def _load_merge_table(file_bytes, file_name, sheet_name):
    return load_dataframe(file_bytes, file_name, sheet_name)


def _save_update(state, dataframe, source_name, sheet_name, message):
    update_working_dataset(state, dataframe, source_name, sheet_name)
    state["manipulation_notice"] = message
    st.rerun()


def render_data_manipulation_page(
    dataframe, state, file_bytes, source_name, sheet_name=None
):
    st.markdown("## Data Manipulation")
    st.caption(
        "Edits update the dataset used by dashboard analyses and exports. "
        "Restore the uploaded file to undo all manual edits."
    )

    notice = state.pop("manipulation_notice", None)
    if notice:
        st.success(notice)

    (
        edit_tab,
        add_tab,
        delete_row_tab,
        add_column_tab,
        delete_column_tab,
        merge_tab,
        type_tab,
    ) = st.tabs(
        [
            "Edit cell",
            "Add record",
            "Delete record",
            "Add column",
            "Delete column",
            "Merge tables",
            "Data type",
        ]
    )

    with edit_tab:
        with st.form("edit_dataset_cell"):
            row_input = st.text_input(
                "Row # or CSV index", placeholder="e.g. 0 or 101"
            )
            column = st.selectbox("Column", list(dataframe.columns))
            new_value = st.text_input("New value", placeholder="Enter updated value")
            submitted = st.form_submit_button("Update cell", type="primary")
        if submitted:
            try:
                updated = edit_cell(dataframe, row_input, column, new_value)
                _save_update(
                    state, updated, source_name, sheet_name,
                    f"Updated row {row_input} in '{column}'.",
                )
            except ValueError as error:
                st.error(str(error))

    with type_tab:
        data_types = [
            "Text",
            "Whole number",
            "Decimal number",
            "Date/time",
            "Boolean",
        ]
        with st.form("change_dataset_column_type"):
            column = st.selectbox(
                "Column",
                list(dataframe.columns),
                key="change_type_column",
            )
            series = dataframe[column]
            if pd.api.types.is_bool_dtype(series.dtype):
                current_target = "Boolean"
            elif pd.api.types.is_integer_dtype(series.dtype):
                current_target = "Whole number"
            elif pd.api.types.is_numeric_dtype(series.dtype):
                current_target = "Decimal number"
            elif pd.api.types.is_datetime64_any_dtype(series.dtype):
                current_target = "Date/time"
            else:
                current_target = "Text"
            current_type = str(series.dtype)
            st.caption(f"Current data type: {current_type}")
            data_type = st.selectbox(
                "New data type",
                data_types,
                index=data_types.index(current_target),
                key="change_type_target",
            )
            submitted = st.form_submit_button(
                "Change data type", type="primary"
            )
        if submitted:
            try:
                updated = change_column_data_type(dataframe, column, data_type)
                _save_update(
                    state,
                    updated,
                    source_name,
                    sheet_name,
                    f"Changed '{column}' to {data_type.lower()}.",
                )
            except ValueError as error:
                st.error(str(error))

    with add_tab:
        with st.form("add_dataset_record"):
            values = {}
            columns = st.columns(3)
            for index, name in enumerate(dataframe.columns):
                label = str(name)
                dtype = dataframe[name].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    placeholder = "Enter a number"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    placeholder = "Enter a date or time"
                else:
                    placeholder = f"Enter {label}"
                with columns[index % len(columns)]:
                    values[name] = st.text_input(
                        label,
                        placeholder=placeholder,
                        key=f"add_record_{index}_{label}",
                    )
            submitted = st.form_submit_button("Add record", type="primary")
        if submitted:
            try:
                updated = add_record(dataframe, values)
                _save_update(
                    state, updated, source_name, sheet_name,
                    f"Added a record at row {len(updated) - 1}.",
                )
            except ValueError as error:
                st.error(str(error))

    with delete_row_tab:
        with st.form("delete_dataset_record"):
            row_input = st.text_input(
                "Row # or CSV index", placeholder="e.g. 0 or 101"
            )
            submitted = st.form_submit_button(
                "Delete record", type="primary"
            )
        if submitted:
            try:
                updated = delete_record(dataframe, row_input)
                _save_update(
                    state, updated, source_name, sheet_name,
                    f"Deleted row {row_input}.",
                )
            except ValueError as error:
                st.error(str(error))

    with add_column_tab:
        with st.form("add_dataset_column"):
            column_name = st.text_input("New column name")
            default_value = st.text_input(
                "Default value", placeholder="Value for existing rows"
            )
            submitted = st.form_submit_button("Add column", type="primary")
        if submitted:
            try:
                updated = add_column(dataframe, column_name, default_value)
                _save_update(
                    state, updated, source_name, sheet_name,
                    f"Added column '{column_name.strip()}'.",
                )
            except ValueError as error:
                st.error(str(error))

    with delete_column_tab:
        with st.form("delete_dataset_column"):
            column = st.selectbox(
                "Column to delete", list(dataframe.columns), key="delete_column_name"
            )
            submitted = st.form_submit_button(
                "Delete column", type="primary"
            )
        if submitted:
            try:
                updated = delete_column(dataframe, column)
                _save_update(
                    state, updated, source_name, sheet_name,
                    f"Deleted column '{column}'.",
                )
            except ValueError as error:
                st.error(str(error))

    with merge_tab:
        st.caption(
            "The current dataset is the first table. Upload one or more additional "
            "tables, then merge them using a column shared by every table. Repeated "
            "values in the shared column can create multiple matching rows."
        )
        uploaded_tables = st.file_uploader(
            "Add tables to merge",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            key="merge_tables_uploader",
        )

        if uploaded_tables:
            merge_candidates = []
            try:
                for index, uploaded_table in enumerate(uploaded_tables):
                    file_bytes = uploaded_table.getvalue()
                    safe_name, _ = validate_uploaded_file(
                        file_bytes, uploaded_table.name
                    )
                    sheet_names = get_excel_sheet_names(file_bytes, safe_name)
                    selected_sheet = None
                    if sheet_names:
                        selected_sheet = st.selectbox(
                            f"Sheet in {safe_name}",
                            sheet_names,
                            key=f"merge_sheet_{index}_{safe_name}",
                        )
                    merge_dataframe = _load_merge_table(
                        file_bytes, safe_name.lower(), selected_sheet
                    )
                    if not merge_dataframe.columns.is_unique:
                        st.error(
                            f"'{safe_name}' has duplicate column names and cannot be merged safely."
                        )
                        st.stop()
                    merge_candidates.append(
                        (safe_name, file_bytes, merge_dataframe)
                    )
            except (ValueError, ImportError, OSError, pd.errors.ParserError) as error:
                st.error(f"Could not load a table for merging: {error}")
                st.stop()

            shared_columns = set(dataframe.columns)
            for _, _, merge_dataframe in merge_candidates:
                shared_columns.intersection_update(merge_dataframe.columns)
            shared_columns = sorted(shared_columns, key=str)

            if not shared_columns:
                st.error(
                    "No column is shared by the current dataset and all uploaded tables."
                )
            else:
                with st.form("merge_uploaded_tables"):
                    shared_column = st.selectbox(
                        "Shared column", shared_columns
                    )
                    merge_type = st.selectbox(
                        "Merge type",
                        ["outer", "inner", "left", "right"],
                        format_func=lambda value: {
                            "outer": "Outer — keep all rows",
                            "inner": "Inner — keep matching rows only",
                            "left": "Left — keep all current dataset rows",
                            "right": "Right — keep all rows from the last table",
                        }[value],
                    )
                    submitted = st.form_submit_button(
                        "Merge tables", type="primary"
                    )

                if submitted:
                    try:
                        merged = merge_tables_on_common_column(
                            dataframe,
                            [
                                (name, table)
                                for name, _, table in merge_candidates
                            ],
                            shared_column,
                            how=merge_type,
                        )
                        update_working_dataset(
                            state, merged, source_name, sheet_name
                        )
                        state["manipulation_notice"] = (
                            f"Merged {len(merge_candidates) + 1} tables on "
                            f"'{shared_column}' using {merge_type.title()} merge."
                        )
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

    st.markdown("### Current data preview")
    st.dataframe(dataframe.head(20), hide_index=True)

    if st.button(
        "Restore uploaded dataset",
        key="restore_uploaded_dataset",
        help="Undo all manual edits and restore the selected uploaded file.",
    ):
        restore_uploaded_dataset(state, file_bytes, source_name, sheet_name)
        state["manipulation_notice"] = "Restored the original uploaded dataset."
        st.rerun()
