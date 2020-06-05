from flask_assets import Environment, Bundle


js = Bundle(
    'node_modules/jquery/dist/jquery.slim.min.js',
    'node_modules/popper.js/dist/umd/popper.min.js',
    'node_modules/bootstrap/dist/js/bootstrap.min.js',
    'node_modules/codecopy/umd/codecopy.min.js',
    filters=('jsmin'),
    output='gen/packed.js'
)

css = Bundle(
    'node_modules/codecopy/umd/codecopy.min.css',
    Bundle('custom.scss', filters=('scss')),
    filters=('cssrewrite'),
    output='gen/packed.css'
)

assets = Environment()
assets.register('js_all', js)
assets.register('css_all', css)
