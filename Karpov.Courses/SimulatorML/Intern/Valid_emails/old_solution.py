# Code to optimize:
# -----------------

import re
from typing import List


def valid_emails(strings: List[str]) -> List[str]:
    '''Take list of potential emails and returns only valid ones'''

    valid_email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    def is_valid_email(email: str) -> bool:
        return bool(re.fullmatch(valid_email_regex, email))

    emails = []
    for email in strings:
        if is_valid_email(email):
            emails.append(email)

    return emails