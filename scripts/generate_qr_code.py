"""Script to generate QR code with logo for this repository."""

from pathlib import Path
import qrcode
from PIL import Image

project_link = Path(__file__).parents[1]
logo_link = project_link / "media" / "logo.png"
url = "https://github.com/danielsanjosepro/hand_manipulation"
save_path = project_link / "media" / "qr_code.png"

logo = Image.open(logo_link)

# taking base width
basewidth = 150

# adjust image size
wpercent = basewidth / float(logo.size[0])
hsize = int(float(logo.size[1]) * float(wpercent))
logo = logo.resize((basewidth, hsize), Image.ADAPTIVE)
QRcode = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)

# taking url or text

# adding URL or text to QRcode
QRcode.add_data(url)

# generating QR code
QRcode.make()

# adding color to QR code
QRimg = QRcode.make_image(fill_color="black", back_color="white").convert("RGB")

# set size of QR code
pos = ((QRimg.size[0] - logo.size[0]) // 2, (QRimg.size[1] - logo.size[1]) // 2)
QRimg.paste(logo, pos)

# save the QR code generated
QRimg.save(save_path)

print(f"QR code generated ahd saved at {save_path}.")
