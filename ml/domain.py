from typing import List, Tuple
import re
import yaml

from globals import get_model_type, SRC_FOLDER

# global data for machine learning, initialised ONCE
# TODO: remove global variables
controllable_params: List = []
context: List = []
configs: List[Tuple[yaml.YAMLObject, str]] = []
current_domain: List = []


def read_and_filter_config(config_path, exclude_patterns: List[str]):
    number_with_unit_pattern = re.compile(r'^\s*(-?\d+(\.\d+)?)(\s*[a-zA-Z%]*)$')
    compiled_exclude_patterns = [re.compile(pattern) for pattern in exclude_patterns]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    filtered_data = []

    def traverse(node, path=''):
        if isinstance(node, bool):
            return
        
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else key
                traverse(value, new_path)

        elif isinstance(node, list):
            for index, item in enumerate(node):
                new_path = f"{path}[{index}]"
                traverse(item, new_path)

        elif isinstance(node, str):
            match = number_with_unit_pattern.match(node)
            if match:

                if '.' in match.group(0):
                    number = float(match.group(1))
                    vartype = 'continuous'
                    search_space = (0, 1) # only one float here
                else:
                    number = int(match.group(1))
                    vartype = 'discrete'
                    search_space = tuple([max(0, (number // 10) * 7), (number // 10) * 13 + 5])

                unit = match.group(3).strip()
                full_path = f"{path}|{unit}" if unit else path
                if not any(pattern.search(full_path) for pattern in compiled_exclude_patterns):
                    filtered_data.append({'name': full_path, 'type': vartype, 'bounds': search_space,
                                           'config idx': len(configs), 'default value': number
                                        })

        elif isinstance(node, (int, float)):
            if not any(pattern.search(path) for pattern in compiled_exclude_patterns):
                if isinstance(node, float):
                    vartype = 'continuous'
                    search_space = (0, 1) # only one float here
                else:
                    vartype = 'discrete'
                    search_space = tuple([max(0, (node // 10) * 7), (node // 10) * 13 + 5])

                filtered_data.append({'name': path, 'type': vartype, 'bounds': search_space,
                                       'config idx': len(configs), 'default value': node
                                    })

    traverse(config)
    configs.append(tuple([config, config_path]))
    return filtered_data

# add numeric params to global list
def read_configs(cfgs: List[str]):
    for cfg in cfgs:
        extracted_data = read_and_filter_config(config_path=cfg, exclude_patterns=["port", "Port", "address", "maxRecvMsgSize", "maxSendMsgSize", "hostConfig.Memory"])
        controllable_params.extend(extracted_data)

    global current_domain
    current_domain = controllable_params

def cut_domain(indices: List[int]):
    global current_domain
    current_domain = [current_domain[i] for i in indices]

def restore_domain():
    global current_domain
    current_domain = controllable_params
    

def apply(X: List):
    print(X)

    for i in range(len(X)):
        index = current_domain[i]['config idx']
        name = current_domain[i]['name']
        vartype = current_domain[i]['type']

        # fill config path based on name
        internal_path = name.split('.')
        parts = internal_path[-1].split('|')
        assert(len(parts) > 0 and len(parts) <= 2)
        internal_path[-1] = parts[0]

        cfg = configs[index][0]
        # спускаемся до РОДИТЕЛЯ нужного ключа, чтобы потом записать parent[key] = ...
        # (раньше тут было `link = link[...]` + `link = X[i]`, что не модифицировало dict)
        parent = cfg[internal_path[0]]
        for j in range(1, len(internal_path) - 1):
            parent = parent[internal_path[j]]
        key = internal_path[-1]

        if len(parts) == 1:
            parent[key] = X[i]
        else:
            if vartype == 'continuous':
                value = float(X[i])
            elif vartype == 'discrete':
                value = int(X[i])
            else:
                value = X[i]
            parent[key] = f"{value}{parts[1]}"

    for cfg, path in configs:
        with open(path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

def transform_domain(domain: List) -> List:
    """
    Transform the domain to a model-specific format
    """
    model_type = get_model_type()
    if model_type == 'BO':
        transformed_domain = []
        for item in domain:
            new_item = item.copy()
            if new_item['type'] == 'continuous':
                new_item['domain'] = item['bounds']
                pass
            elif new_item['type'] == 'discrete':
                new_item['domain'] = list(i for i in range(item['bounds'][0], item['bounds'][1] + 1))
                pass
            else:
                raise ValueError(f"Unknown type: {new_item['type']}")
            new_item.pop('bounds', None)
            transformed_domain.append(new_item)
    else:
        raise ValueError(f"Domain transformation not supported for: {model_type}")

    return transformed_domain

# TODO: do not return domain, just ctx
def fix_domain(domain: List, importance_idx: List[str], best_X) -> Tuple[List, List]:
    """
    Extract context from domain
    """
    importance_idx = set([int(idx) for idx in importance_idx])
    ctx = dict()
    for i, item in enumerate(domain):
        if i not in importance_idx:
            ctx[item['name']] =  best_X[i]

    return domain, ctx