import ipywidgets as widgets

exp_names_ = ['ONE_FLOW_freeze_all', 'ONE_FLOW_freeze_wo2', 'ONE_FLOW_freeze_wo1', 'SOME_EXP']

exp_name_ = widgets.ToggleButtons(options=exp_names_,
                                description='exp name:', disabled=False)

domain_ = widgets.ToggleButtons(options=['H36M', 'MPII'],
                                description='domain:', disabled=False)

mode_ = widgets.ToggleButtons(options=['train', 'val'], 
                                description='mode:', disabled=False)

loss_map_ = {'Pose Regression':'pr', 'Domain Confusion':'dd'}
loss_type_ = widgets.ToggleButtons(options=['Pose Regression', 'Domain Confusion'],
                                description='loss type:', disabled=False)

max_x = 100000
step = 50
slices_ = widgets.SelectionRangeSlider(options=list((range(0, max_x+step)))[0::step], 
                                      index=[0,max_x//step],
                                description='x range: ',disabled=False)

do_aver_ = widgets.ToggleButtons(options=['do averaging', 'raw batch losses'], 
                                description='aver:', disabled=False)