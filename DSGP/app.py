from flask import Flask, render_template, request, jsonify
import pandas as pd
import difflib
app = Flask(__name__)
# Load laptop data from CSV
laptops_df = pd.read_csv("laptops_cleaned.csv")

@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('page1Select.html')

@app.route('/select_dropdown', methods=['POST'])
def select_dropdown():
    return render_template('recommendationPage2.html')

@app.route('/select_sentence', methods=['POST'])
def select_sentence():
    return render_template('sentence.html')

@app.route('/advance_rec_drop', methods=['GET', 'POST'])
def advance_rec_drop():
    if request.method == 'POST':
        price = request.form.get("priceRange")
        ram = request.form.get("ram")
        gpu = request.form.get("gpu")
        advance_rec_dropdown(price, ram, gpu)
        return render_template('recommendationPage2.html', price=price)


@app.route('/sample', methods=['GET', 'POST'])
def sample_fun():
    if request.method == 'POST':
        print("hello world")

    return render_template('recommendationPage1.html')


def advance_rec_dropdown(price, ram, gpu):
    recommendation = recommend_laptop_dropdown(price, ram, gpu)
    print(recommendation)

def recommend_laptop_dropdown(price, ram, gpu):
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

    #print(len(filtered_laptops))
    # Sort laptops by price and return the top recommendation
    if not filtered_laptops.empty:
        return filtered_laptops.sort_values(by='Price_euros').iloc[0]
        #return filtered_laptops
    #if there are no results for the given inputs then the system will use the similar GPUs
    else:
        similar_gpus = find_similar_gpu(gpu, gpus_list)
        recommendation = recommend_laptop_nlp(price, ram, gpu, similar_gpus)
        return recommendation
        #return "No laptops match the specified criteria."


gpus_list = laptops_df['Gpu'].unique().tolist()
def find_similar_gpu(user_gpu, gpus_list, threshold=0.8):
    """
    Find GPUs in the list that are similar to the user-specified GPU.
    """
    similar_gpus = []
    words = user_gpu.split()
    for gpu in gpus_list:
        similarity = difflib.SequenceMatcher(None, user_gpu.lower(), gpu.lower()).ratio()
        if len(words) == 1:
            if similarity >= 0.3:
                similar_gpus.append(gpu)
        elif len(words) == 2:
            if similarity >= 0.5:
                similar_gpus.append(gpu)
        elif len(words) == 3:
            if similarity >= 0.8:
                similar_gpus.append(gpu)
        else:
            if similarity >= 0.9:
                similar_gpus.append(gpu)
    return similar_gpus

def recommend_laptop_nlp(price, ram, gpu, similar_gpus):
    """Recommend a laptop based on price, RAM, and GPU."""
    # Filter laptops based on user input
    filtered_laptops = laptops_df
    if price:
        filtered_laptops = filtered_laptops[filtered_laptops['Price_euros'] <= price]
    if ram:
        filtered_laptops = filtered_laptops[filtered_laptops['Ram'] >= ram]
    # if gpu:
    #     filtered_laptops = filtered_laptops[filtered_laptops['Gpu'] == gpu]
    if gpu:
        if not len(similar_gpus) == 0:
            filtered_laptops = filtered_laptops[filtered_laptops['Gpu'].isin(similar_gpus)]
        else:
            similar_gpus1 = []
            words = gpu.split()
            user_gpu = words[0]
            for gpu in gpus_list:
                similarity = difflib.SequenceMatcher(None, user_gpu.lower(), gpu.lower()).ratio()
                if similarity >= 0.4:
                    similar_gpus1.append(gpu)
            filtered_laptops = filtered_laptops[filtered_laptops['Gpu'].isin(similar_gpus1)]
    #print(len(filtered_laptops))
    # Sort laptops by price and return the top recommendation
    if not filtered_laptops.empty:
        return filtered_laptops.sort_values(by='Price_euros').iloc[0]
        #return filtered_laptops
    else:
        return "No laptops match the specified criteria."

if __name__ == '__main__':
    app.run()