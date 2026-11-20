"""Build the Shopify version of the ring studio from index.html.

  python3 tools/build_shopify.py OUT_DIR [PRICES_JSON]

Writes a theme-shaped folder into OUT_DIR:
  theme/sections/pkh-ring-studio.liquid     the studio as one theme section (styles isolated in a shadow root)
  theme/templates/product.ring-studio.json  a product template that holds only that section
  theme/assets/pkh-ring-*.png|jpg           every image, renamed flat (theme assets cannot sit in folders)
With PRICES_JSON (kept outside this public repo) it also writes:
  product-import.csv                        the draft product and its 60 priced variants, for Shopify's product import
  ../shop-preview.html next to index.html   a local test page with a stand-in store header and cart

index.html stays the single source: edit it, then run this again.
"""
import csv, json, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
out = Path(sys.argv[1]); prices = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
html = (REPO / 'index.html').read_text()

css = re.search(r'<style>\n(.*?)</style>', html, re.S).group(1)
main = re.search(r'<main class="studio">.*?</main>', html, re.S).group(0)
modal = re.search(r'<div class="modal-bg" id="modalBg">.*?\n</div>\n', html, re.S).group(0)
js = re.search(r'<script>\n(.*?)</script>', html, re.S).group(1)
fonts = '\n'.join(re.findall(r'<link [^>]*fonts\.g[^>]*>', html))

def swap(text, a, b):
    assert text.count(a) == 1, a
    return text.replace(a, b)
# the studio's styles live in a shadow root: the store's theme cannot reach in, and these cannot leak out
css = swap(css, ':root{', ':host{')
css = swap(css, 'html,body{margin:0;padding:0}\n', '')
css = swap(css, 'body{background:var(--paper);', ':host{all:initial; display:block; background:var(--paper);')
css = swap(css, '.modal-bg{position:fixed; inset:0;', '.modal-bg{position:fixed; inset:0; z-index:2147483000;')
css = swap(css, '.studio{display:grid; grid-template-columns:minmax(0,1.3fr) minmax(380px,1fr); min-height:calc(100vh - 70px)}',
                '.studio{display:grid; grid-template-columns:minmax(0,1.3fr) minmax(380px,1fr); max-width:1500px; margin:0 auto}')
js = swap(js, 'const ROOT = document;', 'const ROOT = __root;')
js = swap(js, 'const SHOP = null;', 'const SHOP = __shop;')
for part in (css, main, modal, js):
    assert '{%' not in part and '{{' not in part and '</script' not in part and '</template' not in part

# ---- images, flat names
files = []
adir = out / 'theme' / 'assets'; adir.mkdir(parents=True, exist_ok=True)
for kind, pattern in (('stones', '*.png'), ('bottom', '*.png'), ('rings', '*.*')):
    for metal in ('gold', 'rose', 'silver'):
        for f in sorted((REPO / 'img' / kind / metal).glob(pattern)):
            files.append((f, f'{kind}-{metal}-{f.name}'))
files.append((REPO / 'img/bottom/shadow.png', 'bottom-shadow.png'))
for src, name in files: shutil.copyfile(src, adir / ('pkh-ring-' + name))
names = [n for _, n in files]

BOOT = """(function(){
  var id = %(id)s;
  var host = document.getElementById('pkh-ring-' + id);
  if (!host || host.shadowRoot) return;
  var __shop = JSON.parse(document.getElementById('pkh-ring-data-' + id).textContent);
  var __root = host.attachShadow({mode:'open'});
  __root.appendChild(document.getElementById('pkh-ring-tpl-' + id).content.cloneNode(true));
  if (__shop.heading) __root.querySelector('h1').textContent = __shop.heading;
  if (__shop.lede) __root.querySelector('.lede').textContent = __shop.lede;
%(raw0)s
%(js)s
%(raw1)s
})();"""
def page(id_expr, id_text, top, data, raw0, raw1):
    return f"""{fonts}
<div class="pkh-ring-studio" id="pkh-ring-{id_text}" style="--top:{top}px"></div>
<script type="application/json" id="pkh-ring-data-{id_text}">{data}</script>
<template id="pkh-ring-tpl-{id_text}">{raw0}
<style>
{css}</style>
{main}
{modal}{raw1}</template>
<script>
{BOOT % dict(id=id_expr, js=js, raw0=raw0, raw1=raw1)}
</script>
"""

liquid_data = """{
  "cartAddUrl": {{ routes.cart_add_url | append: '.js' | json }},
  "cartUrl": {{ routes.cart_url | json }},
  "currency": {{ cart.currency.iso_code | json }},
  "heading": {{ section.settings.heading | json }},
  "lede": {{ section.settings.lede | json }},
  "naturalGrade": {{ section.settings.natural_grade | json }},
  "variants": { {%- for v in pkh_product.variants -%}{%- if v.sku != blank -%}{{ v.sku | remove_first: 'PKH-BWR-' | json }}:{"id":{{ v.id }},"price":{{ v.price }}}{%- unless forloop.last -%},{%- endunless -%}{%- endif -%}{%- endfor -%} },
  "assets": { {%- for f in pkh_files -%}{{ f | json }}:{{ 'pkh-ring-' | append: f | asset_url | json }}{%- unless forloop.last -%},{%- endunless -%}{%- endfor -%} }
}"""
schema = {
  "name": "Birthstone Ring Studio", "tag": "section", "class": "pkh-ring-studio-section",
  "settings": [
    {"type": "product", "id": "product", "label": "Ring product", "info": "The product whose variants hold the prices. Leave empty on a product template: the page's own product is used."},
    {"type": "text", "id": "heading", "label": "Heading", "default": "Birthstone Ring"},
    {"type": "text", "id": "lede", "label": "Line under the heading", "default": "Made to order in 14k. A birthstone for each wrap, and a diamond to finish."},
    {"type": "text", "id": "natural_grade", "label": "Natural diamond quality", "info": "Added to the order for natural diamonds.", "default": "F-G colour, SI1 clarity"},
    {"type": "range", "id": "sticky_top", "label": "Space for a sticky header", "min": 0, "max": 200, "step": 4, "default": 0, "unit": "px"}
  ],
  "presets": [{"name": "Birthstone Ring Studio"}]
}
section = ("{%- comment -%}\n  PK HART - Birthstone Ring Studio. Built by tools/build_shopify.py from the studio's index.html:\n"
  "  change it there and rebuild, not here. Prices are not in this file: they are the prices of the\n"
  "  ring product's variants, matched by SKU (PKH-BWR-<N|L><030|070|100>-<wraps>W-<A|B>).\n{%- endcomment -%}\n"
  "{%- assign pkh_product = section.settings.product | default: product -%}\n"
  "{%- assign pkh_files = \"" + ','.join(names) + "\" | split: ',' -%}\n"
  + page('{{ section.id | json }}', '{{ section.id }}', '{{ section.settings.sticky_top }}', liquid_data, '{% raw %}', '{% endraw %}')
  + "{% schema %}\n" + json.dumps(schema, indent=2) + "\n{% endschema %}\n")
(out / 'theme/sections').mkdir(parents=True, exist_ok=True); (out / 'theme/templates').mkdir(parents=True, exist_ok=True)
(out / 'theme/sections/pkh-ring-studio.liquid').write_text(section)
(out / 'theme/templates/product.ring-studio.json').write_text(json.dumps({"sections": {"main": {"type": "pkh-ring-studio", "settings": {}}}, "order": ["main"]}, indent=2) + '\n')
print('section', len(section.encode()), 'bytes;', len(names), 'images')

if prices:
    DIAMONDS = [('N', 'natural', 'Natural'), ('L', 'lab', 'Lab-grown')]
    CARATS = [('030', '0.30 ct'), ('070', '0.70 ct'), ('100', '1.00 ct')]
    SIZES = [('A', '4 to 7', 0), ('B', '7.5 to 10', prices['upsize'])]
    rows, variants = [], {}
    for code, key, label in DIAMONDS:
        for cid, clabel in CARATS:
            for w in range(1, 6):
                for scode, slabel, extra in SIZES:
                    sku = f'{code}{cid}-{w}W-{scode}'; price = prices[key][cid][w - 1] + extra
                    variants[sku] = {'id': 40000000000 + len(variants), 'price': price * 100}
                    rows.append((f'{label} {clabel}', str(w), slabel, 'PKH-BWR-' + sku, price))
    head = ['Handle', 'Title', 'Body (HTML)', 'Vendor', 'Type', 'Tags', 'Published', 'Option1 Name', 'Option1 Value', 'Option2 Name', 'Option2 Value',
            'Option3 Name', 'Option3 Value', 'Variant SKU', 'Variant Inventory Policy', 'Variant Fulfillment Service', 'Variant Price',
            'Variant Requires Shipping', 'Variant Taxable', 'Status']
    body = ('<p>A wrap ring made to order in 14k gold: a birthstone for each wrap, and a diamond to finish.</p>'
            '<p>Natural diamonds are F-G colour, SI1 clarity. Lab-grown diamonds are available.</p>')
    with open(out / 'product-import.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(head)
        for i, (o1, o2, o3, sku, price) in enumerate(rows):
            first = i == 0
            w.writerow(['birthstone-wrap-ring', 'Birthstone Wrap Ring' if first else '', body if first else '', 'PK Hart' if first else '', 'Ring' if first else '',
                        'ring-studio' if first else '', 'FALSE' if first else '', 'Bottom diamond' if first else '', o1, 'Wraps' if first else '', o2,
                        'Ring size' if first else '', o3, sku, 'continue', 'manual', f'{price:.2f}', 'TRUE', 'TRUE', 'draft' if first else ''])
    mock = {'cartAddUrl': '/cart/add.js', 'cartUrl': '#bag', 'currency': 'USD', 'heading': 'Birthstone Ring',
            'lede': 'Made to order in 14k. A birthstone for each wrap, and a diamond to finish.', 'naturalGrade': 'F-G colour, SI1 clarity',
            'variants': variants,
            'assets': {n: 'img/' + n.replace('-', '/', 2) if n != 'bottom-shadow.png' else 'img/bottom/shadow.png' for n in names}}
    preview = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shop preview (local test only)</title>
<!-- A stand-in store page: a sticky header and deliberately hostile theme styles, to prove the studio is isolated. -->
<style>
body{margin:0; font-family:Comic Sans MS, cursive; color:purple; letter-spacing:.3em; text-transform:lowercase}
header.theme{position:sticky; top:0; height:72px; background:#222; color:#fff; display:flex; align-items:center; padding:0 24px; z-index:100}
button{background:red !important; border-radius:30px; padding:40px} img{max-width:20px; border:4px solid lime} h1{font-size:90px; color:red}
select{font-size:40px} p{margin:60px} *{box-sizing:content-box}
</style>
<header class="theme">Stand-in store header (sticky, 72px)</header>
<script>window.__cart = []; const realFetch = window.fetch;
window.fetch = (url, opt) => String(url).includes('/cart/add') ? (window.__cart.push(JSON.parse(opt.body)), Promise.resolve(new Response('{}', {status:200}))) : realFetch(url, opt);</script>
""" + page('"test"', 'test', 72, json.dumps(mock), '', '') + '<footer style="height:300px;background:#eee">Stand-in store footer</footer></html>\n'
    (REPO / 'shop-preview.html').write_text(preview)
    print('csv rows', len(rows), '; preview written')
