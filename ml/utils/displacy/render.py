# coding: utf8
from __future__ import unicode_literals

import uuid

from ml.utils.displacy.templates import *
from spacy.util import minify_html, escape_html

DEFAULT_LANG = "en"
DEFAULT_DIR = "ltr"


class DependencyRenderer(object):
    """Render dependency parses as SVGs."""

    style = "dep"

    def __init__(self, options={}):
        """Initialise dependency renderer.

        options (dict): Visualiser-specific options (compact, word_spacing,
            arrow_spacing, arrow_width, arrow_stroke, distance, offset_x,
            color, bg, font)
        """
        self.compact = options.get("compact", False)
        self.word_spacing = options.get("word_spacing", 20)
        self.word_font_size = options.get("word_font_size", 14)
        self.arrow_spacing = options.get("arrow_spacing", 12 if self.compact else 20)
        self.arrow_width = options.get("arrow_width", 6 if self.compact else 10)
        self.arrow_stroke = options.get("arrow_stroke", 2)
        self.distance = options.get("distance", 100 if self.compact else 175)
        self.offset_x = options.get("offset_x", 50)
        # self.offset_y = options.get("offset_y", 50)
        self.color = options.get("color", "#000000")
        self.bg = options.get("bg", "#ffffff")
        self.font = options.get("font", "Arial")
        self.arrow_font_size = options.get("arrow_font_size", 10)  # in pixels
        # self.height = options.get("height", None)
        self.direction = DEFAULT_DIR
        self.lang = DEFAULT_LANG

    ##############
    # MY VERSION #
    ##############
    def render_pair(self, gold_parse, pred_parse, page=False, minify=False, options=None):
        """Render complete markup for a gold/pred pair

        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (unicode): Rendered SVG or HTML markup.
        """
        # Create a random ID prefix to make sure parses don't receive the
        # same ID, even if they're identical

        settings = gold_parse.get("settings", {})
        self.direction = settings.get("direction", DEFAULT_DIR)
        self.lang = settings.get("lang", DEFAULT_LANG)
        svg = self.render_pair_svg(gold_parse["words"], gold_parse["arcs"], pred_parse["words"], pred_parse["arcs"])
        rendered = [svg]
        if page:
            content = "".join([TPL_FIGURE.format(content=svg) for svg in rendered])
            markup = TPL_PAGE.format(content=content, lang=self.lang, dir=self.direction)
        else:
            markup = "".join(rendered)
        if minify:
            return minify_html(markup)
        return markup

    def render_pair_svg(self, gold_words, gold_arcs, pred_words, pred_arcs):
        """ Render the gold parse. Very similar to regular render_svg, but tweaked
        """
        # self.compact = options.get("compact", False)
        # self.word_spacing = options.get("word_spacing", 45)
        # self.arrow_spacing = options.get("arrow_spacing", 12 if self.compact else 20)
        # self.arrow_width = options.get("arrow_width", 6 if self.compact else 10)
        # self.arrow_stroke = options.get("arrow_stroke", 2)
        # self.distance = options.get("distance", 150 if self.compact else 175)
        # self.offset_x = options.get("offset_x", 50)
        # # self.offset_y = options.get("offset_y", 50)
        # self.color = options.get("color", "#000000")
        # self.bg = options.get("bg", "#ffffff")
        # self.font = options.get("font", "Arial")
        # # self.height = options.get("height", None)
        # self.direction = DEFAULT_DIR
        # self.lang = DEFAULT_LANG
        if not self.compact:
            raise NotImplementedError()
        id_prefix = uuid.uuid4().hex
        levels = self.get_levels(gold_arcs)
        highest_level = len(levels)
        gold_arrows_height = highest_level * (self.arrow_font_size + self.arrow_spacing + self.arrow_stroke) + 20
        self.height = gold_arrows_height

        # width = offset_x + len(words) * self.distance

        # offset_y = self.distance / 2 * highest_level + self.arrow_stroke

        # self.id = render_id
        # Render words with gold and pred pos tags
        self.width = self.offset_x + len(gold_words) * self.distance

        # Render the gold arcs on top
        arcs = []
        for i, arc in enumerate(gold_arcs):
            start, end, label, direction, correct = arc["start"], arc["end"], arc["label"], arc["dir"], arc["correct"]
            level = levels.index(end - start) + 1
            x_start = self.offset_x + start * self.distance + self.arrow_spacing
            if self.direction == "rtl":
                x_start = self.width - x_start
            y = gold_arrows_height
            x_end = (
                self.offset_x
                + (end - start) * self.distance
                + start * self.distance
                - self.arrow_spacing * (highest_level - level) / 4
            )
            if self.direction == "rtl":
                x_end = self.width - x_end
            y_curve = gold_arrows_height - level * self.distance / 2
            if self.compact:
                y_curve = gold_arrows_height - level * (
                    self.arrow_font_size + self.arrow_spacing + self.arrow_stroke
                )  # * self.distance / 6
            if y_curve == 0 and len(levels) > 5:
                y_curve = -self.distance

            arc = self.get_arc(x_start, y, y_curve, x_end)
            label_side = "right" if self.direction == "rtl" else "left"
            if correct == "wrong-label":
                strokeColor = "#9c27b0"
                strokeWidth = max(5, 2 * self.arrow_stroke)
            elif correct:
                strokeColor = "currentColor"
                strokeWidth = self.arrow_stroke
            else:
                strokeColor = "#f44336"
                strokeWidth = strokeWidth = max(5, 2 * self.arrow_stroke)
            arrowhead = self.get_arrowhead(direction, x_start, y, x_end, 2 * strokeWidth)
            arcs.append(
                TPL_DEP_ARCS_GP.format(
                    id=id_prefix,
                    i=i,
                    stroke=strokeWidth,
                    textOffset=-strokeWidth,
                    head=arrowhead,
                    label=label,
                    label_side=label_side,
                    arc=arc,
                    arrowFontSize=self.arrow_font_size,
                    strokeColor=strokeColor,
                )
            )

        self.height += 10
        words = []
        for i, (g, p) in enumerate(zip(gold_words, pred_words)):
            y = self.height + self.word_font_size
            x = self.offset_x + i * self.distance
            if self.direction == "rtl":
                x = self.width - x
            html_text = escape_html(g["text"])
            tagColor = "currentColor" if g["correct"] else "#f44336"
            # print(g["correct"])
            words.append(
                TPL_DEP_WORD_GP.format(
                    text=html_text,
                    gtag=g["tag"],
                    ptag=p["tag"],
                    tagColor=tagColor,
                    x=x,
                    y=y,
                    fontSize=f"{self.word_font_size}px",
                    fontSep=f"{2*self.word_font_size}px",
                    tagFontWeight=300 if g["correct"] else 800,
                )
            )
        self.height += 5.5 * self.word_font_size

        # Render pred arcs
        levels = self.get_levels(pred_arcs)
        highest_level = len(levels)
        pred_arrows_height = (
            highest_level * (self.arrow_font_size + self.arrow_spacing + self.arrow_stroke) + self.arrow_font_size + 10
        )
        for i, arc in enumerate(pred_arcs):
            i += len(gold_arcs)
            start, end, label, direction, correct = arc["start"], arc["end"], arc["label"], arc["dir"], arc["correct"]
            level = levels.index(end - start) + 1
            x_start = self.offset_x + start * self.distance + self.arrow_spacing
            if self.direction == "rtl":
                x_start = self.width - x_start
            y = self.height
            x_end = (
                self.offset_x
                + (end - start) * self.distance
                + start * self.distance
                - self.arrow_spacing * (highest_level - level) / 4
            )
            if self.direction == "rtl":
                x_end = self.width - x_end
            y_curve = y + level * self.distance / 2
            if self.compact:
                y_curve = y + level * (
                    self.arrow_font_size + self.arrow_spacing + self.arrow_stroke
                )  # * self.distance / 6
            if y_curve == 0 and len(levels) > 5:
                y_curve = self.distance

            arc = self.get_arc(x_start, y, y_curve, x_end)
            label_side = "right" if self.direction == "rtl" else "left"
            if correct == "wrong-label":
                strokeColor = "#9c27b0"
                strokeWidth = max(5, 2 * self.arrow_stroke)
            elif correct:
                strokeColor = "currentColor"
                strokeWidth = self.arrow_stroke
            else:
                strokeColor = "#f44336"
                strokeWidth = strokeWidth = max(5, 2 * self.arrow_stroke)
            arrowhead = self.get_arrowhead(direction, x_start, y, x_end, 2 * strokeWidth, up=True)
            arcs.append(
                TPL_DEP_ARCS_GP.format(
                    id=id_prefix,
                    i=i,
                    stroke=strokeWidth,
                    head=arrowhead,
                    label=label,
                    label_side=label_side,
                    arc=arc,
                    arrowFontSize=self.arrow_font_size,
                    strokeColor=strokeColor,
                    textOffset=0.75 * strokeWidth + self.arrow_font_size,
                )
            )
        self.height += pred_arrows_height

        # self.height = 15 + gold_arrows_height + 5 * self.word_font_size
        content = "".join(words) + "".join(arcs)
        return TPL_DEP_SVG.format(
            id=id_prefix,
            width=self.width,
            height=self.height,
            color=self.color,
            bg=self.bg,
            font=self.font,
            content=content,
            dir=self.direction,
            lang=self.lang,
        )
        # words = [self.render_word(w["text"], w["tag"], i) for i, w in enumerate(words)]
        # arcs = [self.render_arrow(a["label"], a["start"], a["end"], a["dir"], i) for i, a in enumerate(arcs)]

    # THEIR VERSION
    ###########

    def render(self, parsed, page=False, minify=False):
        """Render complete markup.

        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (unicode): Rendered SVG or HTML markup.
        """
        # Create a random ID prefix to make sure parses don't receive the
        # same ID, even if they're identical
        id_prefix = uuid.uuid4().hex
        rendered = []
        for i, p in enumerate(parsed):
            if i == 0:
                settings = p.get("settings", {})
                self.direction = settings.get("direction", DEFAULT_DIR)
                self.lang = settings.get("lang", DEFAULT_LANG)
            render_id = "{}-{}".format(id_prefix, i)
            svg = self.render_svg(render_id, p["words"], p["arcs"])
            rendered.append(svg)
        if page:
            content = "".join([TPL_FIGURE.format(content=svg) for svg in rendered])
            markup = TPL_PAGE.format(content=content, lang=self.lang, dir=self.direction)
        else:
            markup = "".join(rendered)
        if minify:
            return minify_html(markup)
        return markup

    def render_svg(self, render_id, words, arcs):
        """Render SVG.

        render_id (int): Unique ID, typically index of document.
        words (list): Individual words and their tags.
        arcs (list): Individual arcs and their start, end, direction and label.
        RETURNS (unicode): Rendered SVG markup.
        """
        self.levels = self.get_levels(arcs)
        self.highest_level = len(self.levels)
        self.offset_y = self.distance / 2 * self.highest_level + self.arrow_stroke
        self.width = self.offset_x + len(words) * self.distance
        self.height = self.offset_y + 3 * self.word_spacing
        self.id = render_id
        words = [self.render_word(w["text"], w["tag"], i) for i, w in enumerate(words)]
        arcs = [self.render_arrow(a["label"], a["start"], a["end"], a["dir"], i) for i, a in enumerate(arcs)]
        content = "".join(words) + "".join(arcs)
        return TPL_DEP_SVG.format(
            id=self.id,
            width=self.width,
            height=self.height,
            color=self.color,
            bg=self.bg,
            font=self.font,
            content=content,
            dir=self.direction,
            lang=self.lang,
        )

    def render_word(self, text, tag, i):
        """Render individual word.

        text (unicode): Word text.
        tag (unicode): Part-of-speech tag.
        i (int): Unique ID, typically word index.
        RETURNS (unicode): Rendered SVG markup.
        """
        y = self.offset_y + self.word_spacing
        x = self.offset_x + i * self.distance
        if self.direction == "rtl":
            x = self.width - x
        html_text = escape_html(text)
        return TPL_DEP_WORDS.format(text=html_text, tag=tag, x=x, y=y)

    def render_arrow(self, label, start, end, direction, i):
        """Render individual arrow.

        label (unicode): Dependency label.
        start (int): Index of start word.
        end (int): Index of end word.
        direction (unicode): Arrow direction, 'left' or 'right'.
        i (int): Unique ID, typically arrow index.
        RETURNS (unicode): Rendered SVG markup.
        """
        level = self.levels.index(end - start) + 1
        x_start = self.offset_x + start * self.distance + self.arrow_spacing
        if self.direction == "rtl":
            x_start = self.width - x_start
        y = self.offset_y
        x_end = (
            self.offset_x
            + (end - start) * self.distance
            + start * self.distance
            - self.arrow_spacing * (self.highest_level - level) / 4
        )
        if self.direction == "rtl":
            x_end = self.width - x_end
        y_curve = self.offset_y - level * self.distance / 2
        if self.compact:
            y_curve = self.offset_y - level * self.distance / 6
        if y_curve == 0 and len(self.levels) > 5:
            y_curve = -self.distance
        arrowhead = self.get_arrowhead(direction, x_start, y, x_end)
        arc = self.get_arc(x_start, y, y_curve, x_end)
        label_side = "right" if self.direction == "rtl" else "left"
        return TPL_DEP_ARCS.format(
            id=self.id,
            i=i,
            stroke=self.arrow_stroke,
            head=arrowhead,
            label=label,
            label_side=label_side,
            arc=arc,
            strokeColor="currentColor",
        )

    def get_arc(self, x_start, y, y_curve, x_end):
        """Render individual arc.

        x_start (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        y_curve (int): Y-corrdinate of Cubic Bézier y_curve point.
        x_end (int): X-coordinate of arrow end point.
        RETURNS (unicode): Definition of the arc path ('d' attribute).
        """
        path = "M{x},{y} C{x},{c} {e},{c} {e},{y}".format(x=x_start, y=y, c=y_curve, e=x_end)

        if self.compact:
            path = "M{x},{y} {x},{c} {e},{c} {e},{y}".format(x=x_start, y=y, c=y_curve, e=x_end)
            # A fancier, slightly more legible curve
            d = abs(x_end - x_start)
            c = min(max(5, int(d / 10)), 35)
            # print(y, y_curve)
            if y > y_curve:
                path = (
                    f"M{x_start},{y} L{x_start},{y_curve+c} "
                    f"C{x_start},{y_curve+c} {x_start},{y_curve} {x_start+c},{y_curve} "  # over first curve
                    f"L{x_end-c},{y_curve}  "  # across top
                    f"C{x_end-c},{y_curve} {x_end},{y_curve} {x_end},{y_curve+c} "  # second curve
                    f"L{x_end},{y}"  # to the bottom
                )
            else:
                path = (
                    f"M{x_start},{y} L{x_start},{y_curve-c} "
                    f"C{x_start},{y_curve-c} {x_start},{y_curve} {x_start+c},{y_curve} "  # over first curve
                    f"L{x_end-c},{y_curve}  "  # across top
                    f"C{x_end-c},{y_curve} {x_end},{y_curve} {x_end},{y_curve-c} "  # second curve
                    f"L{x_end},{y}"  # to the bottom
                )
                # print(path)
        return path

    def get_arrowhead(self, direction, x, y, end, arrow_width=None, up=False):
        """Render individual arrow head.

        direction (unicode): Arrow direction, 'left' or 'right'.
        x (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        end (int): X-coordinate of arrow end point.
        RETURNS (unicode): Definition of the arrow head path ('d' attribute).
        """
        arrow_width = arrow_width or self.arrow_width
        if direction == "left":
            pos1, pos2, pos3 = (x, x - 1.25 * arrow_width, x + 1.25 * arrow_width)
        else:
            pos1, pos2, pos3 = (
                end,
                end + 1.25 * arrow_width,
                end - 1.25 * arrow_width,
            )
        if up:
            arrowhead = (
                pos1,
                y - 2,
                pos2,
                y + arrow_width,
                pos3,
                y + arrow_width,
            )
        else:
            arrowhead = (
                pos1,
                y + 2,
                pos2,
                y - arrow_width,
                pos3,
                y - arrow_width,
            )
        return "M{},{} L{},{} {},{}".format(*arrowhead)

    def get_levels(self, arcs):
        """Calculate available arc height "levels".
        Used to calculate arrow heights dynamically and without wasting space.

        args (list): Individual arcs and their start, end, direction and label.
        RETURNS (list): Arc levels sorted from lowest to highest.
        """
        levels = set(map(lambda arc: arc["end"] - arc["start"], arcs))
        return sorted(list(levels))


class EntityRenderer(object):
    """Render named entities as HTML."""

    style = "ent"

    def __init__(self, options={}):
        """Initialise dependency renderer.

        options (dict): Visualiser-specific options (colors, ents)
        """
        colors = {
            "ORG": "#7aecec",
            "PRODUCT": "#bfeeb7",
            "GPE": "#feca74",
            "LOC": "#ff9561",
            "PERSON": "#aa9cfc",
            "NORP": "#c887fb",
            "FACILITY": "#9cc9cc",
            "EVENT": "#ffeb80",
            "LAW": "#ff8197",
            "LANGUAGE": "#ff8197",
            "WORK_OF_ART": "#f0d0ff",
            "DATE": "#bfe1d9",
            "TIME": "#bfe1d9",
            "MONEY": "#e4e7d2",
            "QUANTITY": "#e4e7d2",
            "ORDINAL": "#e4e7d2",
            "CARDINAL": "#e4e7d2",
            "PERCENT": "#e4e7d2",
        }
        colors.update(options.get("colors", {}))
        self.default_color = "#ddd"
        self.colors = colors
        self.ents = options.get("ents", None)
        self.direction = DEFAULT_DIR
        self.lang = DEFAULT_LANG

    def render(self, parsed, page=False, minify=False):
        """Render complete markup.

        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (unicode): Rendered HTML markup.
        """
        rendered = []
        for i, p in enumerate(parsed):
            if i == 0:
                settings = p.get("settings", {})
                self.direction = settings.get("direction", DEFAULT_DIR)
                self.lang = settings.get("lang", DEFAULT_LANG)
            rendered.append(self.render_ents(p["text"], p["ents"], p.get("title")))
        if page:
            docs = "".join([TPL_FIGURE.format(content=doc) for doc in rendered])
            markup = TPL_PAGE.format(content=docs, lang=self.lang, dir=self.direction)
        else:
            markup = "".join(rendered)
        if minify:
            return minify_html(markup)
        return markup

    def render_ents(self, text, spans, title):
        """Render entities in text.

        text (unicode): Original text.
        spans (list): Individual entity spans and their start, end and label.
        title (unicode or None): Document title set in Doc.user_data['title'].
        """
        markup = ""
        offset = 0
        for span in spans:
            label = span["label"]
            start = span["start"]
            end = span["end"]
            entity = escape_html(text[start:end])
            fragments = text[offset:start].split("\n")
            for i, fragment in enumerate(fragments):
                markup += escape_html(fragment)
                if len(fragments) > 1 and i != len(fragments) - 1:
                    markup += "</br>"
            if self.ents is None or label.upper() in self.ents:
                color = self.colors.get(label.upper(), self.default_color)
                ent_settings = {"label": label, "text": entity, "bg": color}
                if self.direction == "rtl":
                    markup += TPL_ENT_RTL.format(**ent_settings)
                else:
                    markup += TPL_ENT.format(**ent_settings)
            else:
                markup += entity
            offset = end
        markup += escape_html(text[offset:])
        markup = TPL_ENTS.format(content=markup, dir=self.direction)
        if title:
            markup = TPL_TITLE.format(title=title) + markup
        return markup
