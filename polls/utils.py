import functools
from nltk.stem import SnowballStemmer

stemmers = {}

def normalize(lang, stem=True, lower=True):
    if stemmers.get(lang) is None:
        stemmers[lang] = SnowballStemmer(lang)

    def decorator_normalize(func):
        _word_stemmer = stemmers[lang]

        @functools.wraps(func)
        def wrapper_stemming(term=None, *args, **kwargs):
            if term is not None:
                if lower:
                    term = term.lower()
                    print(term)
                if stem:
                    term = _word_stemmer.stem(term.lower())
                    print(term)

            return func(term, *args, **kwargs)

        return wrapper_stemming

    return decorator_normalize




def loader(obj, subclass=None, exclude_subclass=True):
    """Return classes of an object.

    Arguments:
        package (package): the package from which you wish to load classes.
        subclass (object): only items that are subclasses of *subclass*.
        exclude_subclass (boolean): False if you want subclass, too.
                                    Defaults to True.
    Returns:
        classes (generator): classes of said package.
    """
    names = (name for name in obj.__dir__() if not name.startswith('__'))
    for name in names:
        item = getattr(obj, name)
        if type(item).__name__ in ('module', 'function'):
            continue
        if subclass is None:
            yield item
        else:
            if issubclass(item, subclass):
                if exclude_subclass:
                    if item is not subclass:
                        if '__metaclass__' not in item.__dict__:
                            yield item
                else:
                    yield item