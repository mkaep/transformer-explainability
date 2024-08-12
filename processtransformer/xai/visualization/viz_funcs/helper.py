
from processtransformer.util.subclassing import BaseSubclasses
from processtransformer.xai.visualization.viz_funcs.base_viz import BaseViz


class BaseVizSubclasses(BaseSubclasses):
    base_class = BaseViz

    @classmethod
    def get_all_subclasses(cls):
        # noinspection PyUnresolvedReferences
        import processtransformer.xai.visualization
        to_check = cls.base_class.__subclasses__()

        return BaseSubclasses.get_subclasses_from_list(to_check)
