"""
Требуется оптимизировать код в 2 раза и соблюсти условия на качество кода.
"""
import re
from typing import List


def valid_emails(strings: List[str]) -> List[str]:
    """Take list of potential emails and returns only valid ones.
    Parameters:
    -----------
    strings: List[str]
        List of potentials emails

    Return:
    -------
    emails: List[str]
        List with valid emails
    """
    valid_email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    pattern = re.compile(valid_email_regex)

    def is_valid_email(email: str) -> bool:
        return bool(pattern.match(email))

    emails = []
    for email in strings:
        if is_valid_email(email):
            emails.append(email)

    return emails
