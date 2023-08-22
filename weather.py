import requests


def what_weather(city):
    # https://wttr.in/:help?lang=ru
    params = {
        'format': 2,  # погода одной строкой
        'M': ''  # скорость ветра в "м/с"
    }
    url = f'http://wttr.in/{city}'
    try:
        result = requests.get(url, params=params)
    except requests.ConnectionError:
        return '<сетевая ошибка>'
    if result.status_code == 200:
        return result.text
    else:
        return '<ошибка на сервере погоды>'
