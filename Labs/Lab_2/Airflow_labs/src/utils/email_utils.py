import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path
from typing import List, Optional

def send_email_gmail(
    subject: str,
    html_body: str,
    to_email: str,
    from_email: str,
    app_password: str,
    attachments: Optional[List[Path]] = None,
    inline_images: Optional[List[Path]] = None,
):
    """
    Sends an HTML email via Gmail SMTP with optional attachments and inline images.
    Inline images will be referenced in HTML as <img src="cid:<basename>">.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    # Attach inline images with Content-ID = basename
    related_images = {}
    if inline_images:
        for p in inline_images:
            p = Path(p)
            cid = make_msgid(domain="inline")[1:-1]  # strip < >
            related_images[p.name] = cid

    # Replace any {{cid:filename}} tokens in html_body with actual cids
    for fname, cid in related_images.items():
        html_body = html_body.replace(f"{{{{cid:{fname}}}}}", f"cid:{cid}")

    msg.add_alternative(html_body, subtype="html")

    # Add the images as related parts
    if inline_images:
        for p in inline_images:
            p = Path(p)
            with open(p, "rb") as f:
                img_data = f.read()
            maintype, subtype = ("image", "png") if p.suffix.lower() == ".png" else ("application", "octet-stream")
            msg.get_payload()[0].add_related(img_data, maintype=maintype, subtype=subtype, cid=related_images[p.name], filename=p.name)

    # Standard attachments
    if attachments:
        for p in attachments:
            p = Path(p)
            with open(p, "rb") as f:
                data = f.read()
            msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=p.name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(from_email, app_password)
        smtp.send_message(msg)
