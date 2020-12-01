# importing necessary modules
from glove_inferencing import glove_model, predict_sentiment
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.view import view_config


# defining parameters of view callable which is called upon request from user to sever like
@view_config(route_name='GloVe', accept='application/json', request_method='POST', renderer='json')
def sentiment_analyse(request):
    """Function predict sentiments of given texts in json format and returns it in json format"""

    # loading json data from request object
    texts = request.json_body["texts"]

    # predicting sentiment of all texts in json
    results = predict_sentiment(glove_model, texts)

    # creating dictionary for text as key & it's predicted sentiment as value
    results = dict(list(zip(texts, results)))

    return results


if __name__ == '__main__':
    # configuring our application
    with Configurator() as config:

        # setting name of route & route(link extension) for an API
        config.add_route('GloVe', '/glove')
        config.scan()

        # starting app
        app = config.make_wsgi_app()

    # setting server and port number
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()


