from speedtoxify import Speedtoxify


def test_simple():
    model = Speedtoxify("original-small")
    model.predict("You are a good person")
