from package_utils.context.entry_point import create_entry_point

from minivit import main
from minivit.context import context

entry_point = create_entry_point(main, context)
