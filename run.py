from argparse import ArgumentParser
from distutils.util import strtobool
from dataclasses import dataclass, field
from enum import Enum
from json import dump
from typing import Dict, List, Optional, Union
import requests
from pyzotero.zotero import Zotero
from pyzotero.zotero_errors import ParamNotPassed, UnsupportedParams
from pathlib import Path
import json 

TOP_DIR = Path(__file__).parent
FAILED_ITEMS_DIR = TOP_DIR

def sanitize_tag(tag: str) -> str:
    """Clean tag by replacing empty spaces with underscore.

    Parameters
    ----------
    tag: str

    Returns
    -------
    str
        Cleaned tag

    Examples
    --------
    >>> sanitize_tag(" Machine Learning ")
    "Machine_Learning"

    """
    return tag.strip().replace(" ", "_")

def read_library_version():
    """
    Reads the library version from the 'since' file and returns it as an integer.
    If the file does not exist or does not include a number, returns 0.
    """
    try:
        with open('since', 'r', encoding='utf-8') as file:
            return int(file.read())
    except FileNotFoundError:
        print("since file does not exist, using library version 0")
    except ValueError:
        print("since file does not include a number, using library version 0")
    return 0

def write_library_version(zotero_client):
    """
    Writes the library version of the given Zotero client to a file named 'since'.

    Args:
        zotero_client: A Zotero client object.

    Returns:
        None
    """
    with open('since', 'w', encoding='utf-8') as file:
        file.write(str(zotero_client.last_modified_version()))
        
class Zotero2ReadwiseError(Exception):
    def __init__(self, message: str):
        self.message = message

        super().__init__(self.message)

@dataclass
class ZoteroItem:
    key: str
    version: int
    item_type: str
    text: str
    annotated_at: str
    annotation_url: str
    comment: Optional[str] = None
    title: Optional[str] = None
    tags: Optional[List[str]] = field(init=True, default=None)
    document_tags: Optional[List[Dict]] = field(init=True, default=None)
    document_type: Optional[int] = None
    annotation_type: Optional[str] = None
    creators: Optional[str] = field(init=True, default=None)
    source_url: Optional[str] = None
    attachment_url: Optional[str] = None
    page_label: Optional[str] = None
    color: Optional[str] = None
    relations: Optional[Dict] = field(init=True, default=None)

    def __post_init__(self):
        # Convert [{'tag': 'abc'}, {'tag': 'def'}] -->  ['abc', 'def']
        if self.tags:
            self.tags = [d_["tag"] for d_ in self.tags]

        if self.document_tags:
            self.document_tags = [d_["tag"] for d_ in self.document_tags]

        # Sample {'dc:relation': ['http://zotero.org/users/123/items/ABC', 'http://zotero.org/users/123/items/DEF']}
        if self.relations:
            self.relations = self.relations.get("dc:relation")
        
        if self.creators:
            et_al = "et al."
            max_length = 1024 - len(et_al)
            creators_str = ", ".join(self.creators)
            if len(creators_str) > max_length:
                # Reset creators_str and find the first n creators that fit in max_length
                creators_str = ""
                while self.creators and len(creators_str) < max_length:
                    creators_str += self.creators.pop() + ", "
                creators_str += et_al
            self.creators = creators_str


    def get_nonempty_params(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v}


def get_zotero_client(
    library_id: str = None, api_key: str = None, library_type: str = "user"
) -> Zotero:
    """Create a Zotero client object from Pyzotero library

    Zotero userID and Key are available

    Parameters
    ----------
    library_id: str
        If not passed, then it looks for `ZOTERO_LIBRARY_ID` in the environment variables.
    api_key: str
        If not passed, then it looks for `ZOTERO_KEY` in the environment variables.
    library_type: str ['user', 'group']
        'user': to access your Zotero library
        'group': to access a shared group library

    Returns
    -------
    Zotero
        a Zotero client object
    """

    if library_id is None:
        try:
            library_id = environ["ZOTERO_LIBRARY_ID"]
        except KeyError:
            raise ParamNotPassed(
                "No value for library_id is found. "
                "You can set it as an environment variable `ZOTERO_LIBRARY_ID` or use `library_id` to set it."
            )

    if api_key is None:
        try:
            api_key = environ["ZOTERO_KEY"]
        except KeyError:
            raise ParamNotPassed(
                "No value for api_key is found. "
                "You can set it as an environment variable `ZOTERO_KEY` or use `api_key` to set it."
            )

    if library_type is None:
        library_type = environ.get("LIBRARY_TYPE", "user")
    elif library_type not in ["user", "group"]:
        raise UnsupportedParams("library_type value can either be 'user' or 'group'.")

    return Zotero(
        library_id=library_id,
        library_type=library_type,
        api_key=api_key,
    )


class ZoteroAnnotationsNotes:
    def __init__(self, zotero_client: Zotero, filter_colors: List[str]):
        self.zot = zotero_client
        self.failed_items: List[Dict] = []
        self._cache: Dict = {}
        self._parent_mapping: Dict = {}
        self.filter_colors: List[str] = filter_colors

    def get_item_metadata(self, annot: Dict) -> Dict:
        data = annot["data"]
        # A Zotero annotation or note must have a parent with parentItem key.
        parent_item_key = data["parentItem"]
        
        if parent_item_key in self._parent_mapping:
            top_item_key = self._parent_mapping[parent_item_key]
            if top_item_key in self._cache:
                return self._cache[top_item_key]
        else:
            parent_item = self.zot.item(parent_item_key)
            top_item_key = parent_item["data"].get("parentItem", None)
            self._parent_mapping[parent_item_key] = (
                top_item_key if top_item_key else parent_item_key
            )

        if top_item_key:
            top_item = self.zot.item(top_item_key)
            data = top_item["data"]
        else:
            top_item = parent_item
            data = top_item["data"]
            top_item_key = data["key"]

        metadata = {
            "title": data["title"],
            # "date": data["date"],
            "tags": data["tags"],
            "document_type": data["itemType"],
            "source_url": top_item["links"]["alternate"]["href"],
            "creators": "",
            "attachment_url": "",
        }
        if "creators" in data:
            metadata["creators"] = [
                creator["firstName"] + " " + creator["lastName"]
                for creator in data["creators"]
            ]
        if "attachment" in top_item["links"] and top_item["links"]["attachment"]["attachmentType"] == "application/pdf":
            metadata["attachment_url"] = top_item["links"]["attachment"]["href"]

        self._cache[top_item_key] = metadata
        return metadata

    def format_item(self, annot: Dict) -> ZoteroItem:
        data = annot["data"]
        item_type = data["itemType"]
        annotation_type = data.get("annotationType")
        metadata = self.get_item_metadata(annot)

        text = ""
        comment = ""
        if item_type == "annotation":
            if annotation_type == "highlight":
                text = data["annotationText"]
                comment = data["annotationComment"]
            elif annotation_type == "note":
                text = data["annotationComment"]
                comment = ""
            else:
                raise NotImplementedError(
                    "Handwritten annotations are not currently supported."
                )
        elif item_type == "note":
            text = data["note"]
            comment = ""
        else:
            raise NotImplementedError(
                "Only Zotero item types of 'note' and 'annotation' are supported."
            )

        if text == "":
            raise ValueError("No annotation or note data is found.")
        return ZoteroItem(
            key=data["key"],
            version=data["version"],
            item_type=item_type,
            text=text,
            annotated_at=data["dateModified"],
            annotation_url=annot["links"]["alternate"]["href"],
            attachment_url=metadata["attachment_url"],
            comment=comment,
            title=metadata["title"],
            tags=data["tags"],
            document_tags=metadata["tags"],
            document_type=metadata["document_type"],
            annotation_type=annotation_type,
            creators=metadata.get("creators"),
            source_url=metadata["source_url"],
            page_label=data.get("annotationPageLabel"),
            color=data.get("annotationColor"),
            relations=data["relations"],
        )

    def format_items(self, annots: List[Dict]) -> List[ZoteroItem]:
        formatted_annots = []
        print(
            f"ZOTERO: Start formatting {len(annots)} annotations/notes...\n"
            f"It may take some time depending on the number of annotations...\n"
            f"A complete message will show up once it's done!\n"
        )
        for annot in annots:
            try:
                if len(self.filter_colors) == 0 or annot["data"]["annotationColor"] in self.filter_colors:
                    formatted_annots.append(self.format_item(annot))
            except:
                self.failed_items.append(annot)
                continue

        finished_msg = "\nZOTERO: Formatting Zotero Items is completed!!\n\n"
        if self.failed_items:
            finished_msg += (
                f"\nNOTE: {len(self.failed_items)} Zotero annotations/notes (out of {len(annots)}) failed to format.\n"
                f"You can run `save_failed_items_to_json()` class method to save those items."
            )
        print(finished_msg)
        return formatted_annots

    def save_failed_items_to_json(self, json_filepath_failed_items: str = None):
        FAILED_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
        if json_filepath_failed_items:
            out_filepath = FAILED_ITEMS_DIR.joinpath(json_filepath_failed_items)
        else:
            out_filepath = FAILED_ITEMS_DIR.joinpath("failed_zotero_items.json")

        with open(out_filepath, "w") as f:
            dump(self.failed_items, f, indent=4)
        print(f"\nZOTERO: Detail of failed items are saved into {out_filepath}\n")

@dataclass
class ReadwiseAPI:
    """Dataclass for ReadWise API endpoints"""

    base_url: str = "https://readwise.io/api/v2"
    highlights: str = base_url + "/highlights/"
    books: str = base_url + "/books/"


class Category(Enum):
    articles = 1
    books = 2
    tweets = 3
    podcasts = 4


@dataclass
class ReadwiseHighlight:
    text: str
    title: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    category: Optional[str] = None
    note: Optional[str] = None
    location: Union[int, None] = 0
    location_type: Optional[str] = "page"
    highlighted_at: Optional[str] = None
    highlight_url: Optional[str] = None

    def __post_init__(self):
        if not self.location:
            self.location = None

    def get_nonempty_params(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v}


class Readwise:
    def __init__(self, readwise_token: str):
        self._token = readwise_token
        self._header = {"Authorization": f"Token {self._token}"}
        self.endpoints = ReadwiseAPI
        self.failed_highlights: List = []

    def create_highlights(self, highlights: List[Dict]) -> None:
        resp = requests.post(
            url=self.endpoints.highlights,
            headers=self._header,
            json={"highlights": highlights},
        )
        if resp.status_code != 200:
            error_log_file = (
                f"error_log_{resp.status_code}_failed_post_request_to_readwise.json"
            )
            with open(error_log_file, "w") as f:
                errors = resp.json()
                mappedErrors = []
                for index, element in enumerate(highlights):
                    error = errors[index]
                    if (len(error.keys()) > 0):
                        error.highlight = element.highlight_url
                        mappedErrors = mappedErrors.append(error)
                        print(error)
                dump(json.dumps(mappedErrors), f)
            raise Zotero2ReadwiseError(
                f"Uploading to Readwise failed with following details:\n"
                f"POST request Status Code={resp.status_code} ({resp.reason})\n"
                f"Error log is saved to {error_log_file} file."
            )

    @staticmethod
    def convert_tags_to_readwise_format(tags: List[str]) -> str:
        return " ".join([f".{sanitize_tag(t.lower())}" for t in tags])

    def format_readwise_note(self, tags, comment) -> Union[str, None]:
        rw_tags = self.convert_tags_to_readwise_format(tags)
        highlight_note = ""
        if rw_tags:
            highlight_note += rw_tags + "\n"
        if comment:
            highlight_note += comment
        return highlight_note if highlight_note else None

    def convert_zotero_annotation_to_readwise_highlight(
        self, annot: ZoteroItem
    ) -> ReadwiseHighlight:

        highlight_note = self.format_readwise_note(
            tags=annot.tags, comment=annot.comment
        )
        if annot.page_label and annot.page_label.isnumeric():
            location = int(annot.page_label)
        else:
            location = 0
        highlight_url = None
        if annot.attachment_url is not None:
            attachment_id = annot.attachment_url.split("/")[-1]
            annot_id = annot.annotation_url.split("/")[-1]
            highlight_url = f'zotero://open-pdf/library/items/{attachment_id}?page={location}%&annotation={annot_id}'
        return ReadwiseHighlight(
            text=annot.text,
            title=annot.title,
            note=highlight_note,
            author=annot.creators,
            category=Category.articles.name
            if annot.document_type != "book"
            else Category.books.name,
            highlighted_at=annot.annotated_at,
            source_url=annot.source_url,
            highlight_url=annot.annotation_url
            if highlight_url is None
            else highlight_url,
            location=location,
        )

    def post_zotero_annotations_to_readwise(
        self, zotero_annotations: List[ZoteroItem]
    ) -> None:
        print(
            f"\nReadwise: Push {len(zotero_annotations)} Zotero annotations/notes to Readwise...\n"
            f"It may take some time depending on the number of highlights...\n"
            f"A complete message will show up once it's done!\n"
        )
        rw_highlights = []
        for annot in zotero_annotations:
            try:
                if len(annot.text) >= 8191:
                    print(
                        f"A Zotero annotation from an item with {annot.title} (item_key={annot.key} and "
                        f"version={annot.version}) cannot be uploaded since the highlight/note is very long. "
                        f"A Readwise highlight can be up to 8191 characters."
                    )
                    self.failed_highlights.append(annot.get_nonempty_params())
                    continue  # Go to next annot
                rw_highlight = self.convert_zotero_annotation_to_readwise_highlight(
                    annot
                )
            except:
                self.failed_highlights.append(annot.get_nonempty_params())
                continue  # Go to next annot
            rw_highlights.append(rw_highlight.get_nonempty_params())
        self.create_highlights(rw_highlights)

        finished_msg = ""
        if self.failed_highlights:
            finished_msg = (
                f"\nNOTE: {len(self.failed_highlights)} highlights (out of {len(self.failed_highlights)}) failed "
                f"to upload to Readwise.\n"
            )

        finished_msg += f"\n{len(rw_highlights)} highlights were successfully uploaded to Readwise.\n\n"
        print(finished_msg)

    def save_failed_items_to_json(self, json_filepath_failed_items: str = None):
        FAILED_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
        if json_filepath_failed_items:
            out_filepath = FAILED_ITEMS_DIR.joinpath(json_filepath_failed_items)
        else:
            out_filepath = FAILED_ITEMS_DIR.joinpath("failed_readwise_items.json")

        with open(out_filepath, "w") as f:
            dump(self.failed_highlights, f)
        print(
            f"{len(self.failed_highlights)} highlights failed to format (hence failed to upload to Readwise).\n"
            f"Detail of failed items are saved into {out_filepath}"
        )

class Zotero2Readwise:
    def __init__(
        self,
        readwise_token: str,
        zotero_key: str,
        zotero_library_id: str,
        zotero_library_type: str = "user",
        include_annotations: bool = True,
        include_notes: bool = False,
        filter_colors: List[str] = [],
        since: int = 0
    ):
        self.readwise = Readwise(readwise_token)
        self.zotero_client = get_zotero_client(
            library_id=zotero_library_id,
            library_type=zotero_library_type,
            api_key=zotero_key,
        )
        self.zotero = ZoteroAnnotationsNotes(self.zotero_client, filter_colors)
        self.include_annots = include_annotations
        self.include_notes = include_notes
        self.since = since

    def get_all_zotero_items(self) -> List[Dict]:
            """
            Retrieves all Zotero items of the specified types (notes and/or annotations) that were modified since the specified date.

            Returns:
            A list of dictionaries representing the retrieved Zotero items.
            """
            items = []
            if self.include_annots:
                items.extend(self.retrieve_all("annotation", self.since))

            if self.include_notes:
                items.extend(self.retrieve_all("note", self.since))

            print(f"{len(items)} Zotero items are retrieved.")

            return items

    def run(self, zot_annots_notes: List[Dict] = None) -> None:
        if zot_annots_notes is None:
            zot_annots_notes = self.get_all_zotero_items()

        formatted_items = self.zotero.format_items(zot_annots_notes)

        if self.zotero.failed_items:
            self.zotero.save_failed_items_to_json("failed_zotero_items.json")

        self.readwise.post_zotero_annotations_to_readwise(formatted_items)
    
    def retrieve_all(self, item_type: str, since: int = 0):
        """
        Retrieves all items of a given type from Zotero Database since a given timestamp.

        Args:
            item_type (str): Either "annotation" or "note".
            since (int): Timestamp in seconds since the Unix epoch. Defaults to 0.

        Returns:
            List[Dict]: List of dictionaries containing the retrieved items.
        """
        if item_type not in ["annotation", "note"]:
            raise ValueError("item_type must be either 'annotation' or 'note'")

        if since == 0:
            print(f"Retrieving ALL {item_type}s from Zotero Database")
        else:
            print(f"Retrieving {item_type}s since last run from Zotero Database")

        print("It may take some time...")
        query = self.zotero_client.items(itemType={item_type}, since=since)
        return self.zotero_client.everything(query)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Markdown files")
    parser.add_argument(
        "readwise_token",
        help="Readwise Access Token (visit https://readwise.io/access_token)",
    )
    parser.add_argument(
        "zotero_key", help="Zotero API key (visit https://www.zotero.org/settings/keys)"
    )
    parser.add_argument(
        "zotero_library_id",
        help="Zotero User ID (visit https://www.zotero.org/settings/keys)",
    )
    parser.add_argument(
        "--library_type",
        default="user",
        help="Zotero Library type ('user': for personal library (default value), 'group': for a shared library)",
    )
    parser.add_argument(
        "--include_annotations",
        type=str,
        default="y",
        help="Include Zotero annotations (highlights + comments) | Options: 'y'/'yes' (default), 'n'/'no'",
    )
    parser.add_argument(
        "--include_notes",
        type=str,
        default="n",
        help="Include Zotero notes | Options: 'y'/'yes', 'n'/'no' (default)",
    )
    parser.add_argument(
        "--filter_color",
        choices=['#ffd400', '#ff6666', '#5fb236', '#2ea8e5', '#a28ae5', '#e56eee', '#f19837', '#aaaaaa'],
        action="append",
        default=[],
        help="Filter Zotero annotations by given color | Options: '#ffd400' (yellow), '#ff6666' (red), '#5fb236' (green), '#2ea8e5' (blue), '#a28ae5' (purple), '#e56eee' (magenta), '#f19837' (orange), '#aaaaaa' (gray)"
    )
    parser.add_argument(
        "--use_since",
        action='store_true',
        help="Include Zotero items since last run"
    )

    args = vars(parser.parse_args())

    # Cast str to bool values for bool flags
    for bool_arg in ["include_annotations", "include_notes"]:
        try:
            args[bool_arg] = bool(strtobool(args[bool_arg]))
        except ValueError:
            raise ValueError(
                f"Invalid value for --{bool_arg}. Use 'n' or 'y' (default)."
            )

    since = read_library_version() if args["use_since"] else 0
    zt2rw = Zotero2Readwise(
        readwise_token=args["readwise_token"],
        zotero_key=args["zotero_key"],
        zotero_library_id=args["zotero_library_id"],
        zotero_library_type=args["library_type"],
        include_annotations=args["include_annotations"],
        include_notes=args["include_notes"],
        filter_colors=args["filter_color"],
        since=since
    )
    zt2rw.run()
    if args["use_since"]:
        write_library_version(zt2rw.zotero_client)