import yagmail

# Your email credentials
sender_email = 'asadmayo42@example.com'
sender_password = 'P@ki$t@n1'

# Initialize yagmail
yag = yagmail.SMTP(user=sender_email, password=sender_password)

# Send email
recipient_email = 'asadmayo42@gmail.com'
subject = 'Test Email from Python'
body = 'This is a test email sent from Python!'

try:
    yag.send(to=recipient_email, subject=subject, contents=body)
    print("Email sent successfully")
except Exception as e:
    print(f"Failed to send email: {e}")