from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def recommend_laptop():
    if request.method == 'POST':
        price = request.form.get("priceRange")
        ram = request.form.get("ram")
        gpu = request.form.get("gpu")

        jnj(price, ram, gpu)
        return render_template('../', price=price)

    # data = request.json
    # price = data['price']
    # ram = data['ram']
    # gpu = data['gpu']
    #
    # # Now you have the user inputs, you can process them as needed
    # print("Price:", price)
    # print("RAM:", ram)
    # print("GPU:", gpu)

    # return jsonify({'success': True})
    return render_template('../UI/recommendationPage2.html')


@app.route('/sample', methods=['GET', 'POST'])
def sample_fun():
    if request.method == 'POST':
        print("hello world")

    return render_template('recommendationPage1.html')


def jnj(price, ram, gpu):
    print(price)
    print(ram)
    print(gpu)


if __name__ == '__main__':
    app.run()
