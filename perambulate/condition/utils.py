import re 
from operator import eq
from operator import ge
from operator import gt
from operator import le
from operator import lt
from operator import ne


def extract_operator(value, default_op: str=None) -> object:
        regex = r"^[><=!]{1,2}"

        mapping = {
            ">=": ge,
            "<=": le,
            ">": gt,
            "<": lt,
            "==": eq,
            "!=": ne,
        }

        try: 
            token = re.findall(regex, value.strip())[0]
            op = mapping[token]
            value = value.replace(token, "")
        except (IndexError, KeyError): 
            if default_op is not None:
                op = mapping[default_op]
            else:
                op = None

        return op, value