from flask import Flask,render_template,request,redirect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
app = Flask(__name__)
import mysql.connector
app.secret_key = 'secret'
con=mysql.connector.connect(host="localhost",user="root",password="",database="registration")
con2=mysql.connector.connect(host="localhost",user="root",password="",database="analytics")
cursor=con.cursor()
cursor2=con2.cursor()


@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/perform_registration',methods=['post'])
def perform_registration():
    name = request.form.get('user_ka_name')
    email = request.form.get('user_ka_id')
    password = request.form.get('user_ka_password')

    cursor.execute("""SELECT * FROM `user` WHERE `email` LIKE  '{}' """.format(email))
    member = cursor.fetchall()
    if len(member)>0:
        return render_template('register.html', message="Email already exists")
    else:
        cursor.execute("""INSERT INTO `user` (`id`,`name`,`email`,`password`)VALUES
        (NULL,'{}','{}','{}')""".format(name,email,password))
        con.commit()
        return render_template('login.html',message="Registration Successful. Kindly login to proceed")


@app.route('/perform_login',methods=['post'])
def perform_login():
    email = request.form.get('user_ka_id')
    password = request.form.get('user_ka_password')

    cursor.execute("""SELECT * FROM `user` WHERE `email` LIKE  '{}' AND `password` LIKE  '{}' """
                   .format(email,password))
    users = cursor.fetchall()
    print(users)
    if len(users)>0:
        return redirect('/profile')
    else:
        return render_template('login.html',message='Incorrect Email/Password')


@app.route('/profile')
def profile():
    return render_template('profile.html')


@app.route('/branch')
def branch():
    cursor2.execute("SELECT * FROM footfalls")
    data = cursor2.fetchall()
    print(type(data))
    df = pd.DataFrame(data, columns=['Male','Female'])

    df.sum().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Male vs Female Proportions')
    chart_filename = 'static/pie_chart.png'  # Save the chart image to a static directory
    plt.savefig(chart_filename)  # Save the pie chart as a PNG image
    return render_template('branch.html',chart_filename=chart_filename)




if __name__ == '__main__':
    app.run(debug=True)