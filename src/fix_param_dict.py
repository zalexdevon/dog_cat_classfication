def fix_param_dict(param_dict):
    new_keys = []
    new_values = []

    for key, value in param_dict.items():
        if isinstance(value, dict) == False:
            new_keys.append(key)
            new_values.append(value)
            continue

        for inner_key, inner_value in value.items():
            new_inner_key = f"{key}__{inner_key}"
            new_keys.append(new_inner_key)
            new_values.append(inner_value)

    return dict(zip(new_keys, new_values))
