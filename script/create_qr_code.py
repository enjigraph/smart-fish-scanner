import qrcode

data="200,100"

qr = qrcode.QRCode(version=4,error_correction=qrcode.constants.ERROR_CORRECT_L,box_size=10,border=4)
qr.add_data(data)
qr.make(fit=True)

img = qr.make_image(fill='black', back_color='white')
img.save('qr_code.png')
img.show()
