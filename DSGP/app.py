from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)
# Load laptop data from CSV
laptops_df = pd.read_csv("laptops_cleaned.csv")


@app.route('/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        price = request.form.get("priceRange")
        ram = request.form.get("ram")
        gpu = request.form.get("gpu")
        advance_rec_dropdown(price, ram, gpu)
        return render_template('recommendationPage1.html', price=price)

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
    return render_template('recommendationPage2.html')


@app.route('/sample', methods=['GET', 'POST'])
def sample_fun():
    if request.method == 'POST':
        print("hello world")

    return render_template('recommendationPage1.html')


def advance_rec_dropdown(price, ram, gpu):
    recommendation = recommend_laptop(price, ram, gpu)
    print(recommendation)

def recommend_laptop(price, ram, gpu):
    price = float(price)
    ram = int(ram)
    """Recommend a laptop based on price, RAM, and GPU."""
    # Filter laptops based on user input
    filtered_laptops = laptops_df
    if price:
        filtered_laptops = filtered_laptops[filtered_laptops['Price_euros'] <= price]
    if ram:
        filtered_laptops = filtered_laptops[filtered_laptops['Ram'] >= ram]
    if gpu:
        filtered_laptops = filtered_laptops[filtered_laptops['Gpu'] == gpu]

    print(len(filtered_laptops))
    # Sort laptops by price and return the top recommendation
    if not filtered_laptops.empty:
        return filtered_laptops.sort_values(by='Price_euros').iloc[0]
        #return filtered_laptops
    else:
        return "No laptops match the specified criteria."


if __name__ == '__main__':
    app.run()
