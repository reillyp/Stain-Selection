#############################################################################
# Generated by PAGE version 4.22
#  in conjunction with Tcl version 8.6
#  Sep 19, 2019 03:53:52 PM CDT  platform: Windows NT
set vTcl(timestamp) ""


if {!$vTcl(borrow)} {

set vTcl(actual_gui_bg) #d9d9d9
set vTcl(actual_gui_fg) #000000
set vTcl(actual_gui_analog) #ececec
set vTcl(actual_gui_menu_analog) #ececec
set vTcl(actual_gui_menu_bg) #d9d9d9
set vTcl(actual_gui_menu_fg) #000000
set vTcl(complement_color) #d9d9d9
set vTcl(analog_color_p) #d9d9d9
set vTcl(analog_color_m) #ececec
set vTcl(active_fg) #000000
set vTcl(actual_gui_menu_active_bg)  #ececec
set vTcl(active_menu_fg) #000000
}

#################################
#LIBRARY PROCEDURES
#


if {[info exists vTcl(sourcing)]} {

proc vTcl:project:info {} {
    set base .top42
    global vTcl
    set base $vTcl(btop)
    if {$base == ""} {
        set base .top42
    }
    namespace eval ::widgets::$base {
        set dflt,origin 0
        set runvisible 1
    }
    namespace eval ::widgets_bindings {
        set tagslist _TopLevel
    }
    namespace eval ::vTcl::modules::main {
        set procs {
        }
        set compounds {
        }
        set projectType single
    }
}
}

#################################
# GENERATED GUI PROCEDURES
#

proc vTclWindow.top42 {base} {
    if {$base == ""} {
        set base .top42
    }
    if {[winfo exists $base]} {
        wm deiconify $base; return
    }
    set top $base
    ###################
    # CREATING WIDGETS
    ###################
    vTcl::widgets::core::toplevel::createCmd $top -class Toplevel \
        -menu "$top.m48" -background {#d9d9d9} -highlightbackground {#d9d9d9} \
        -highlightcolor black 
    wm focusmodel $top passive
    wm geometry $top 802x654+306+124
    update
    # set in toplevel.wgt.
    global vTcl
    global img_list
    set vTcl(save,dflt,origin) 0
    wm maxsize $top 1684 1031
    wm minsize $top 120 1
    wm overrideredirect $top 0
    wm resizable $top 1 1
    wm deiconify $top
    wm title $top "Stain Selection"
    vTcl:DefineAlias "$top" "Toplevel1" vTcl:Toplevel:WidgetProc "" 1
    button $top.but43 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 16 -weight bold} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text EXIT 
    vTcl:DefineAlias "$top.but43" "ButtonExit" vTcl:WidgetProc "Toplevel1" 1
    bind $top.but43 <Button-1> {
        lambda e: btn_exit(e)
    }
    button $top.but45 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text {Select Input Image} 
    vTcl:DefineAlias "$top.but45" "ButtonInputImg" vTcl:WidgetProc "Toplevel1" 1
    bind $top.but45 <ButtonRelease-1> {
        lambda e: btn_select_input_img(e)
    }
    label $top.lab46 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -justify right \
        -text {Enter Lower Threshold  Values (0-255): } 
    vTcl:DefineAlias "$top.lab46" "LabelLowHSV" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab48 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -justify right \
        -text {Enter Upper Threshold Values (0-255):} 
    vTcl:DefineAlias "$top.lab48" "LabelUpperHSV" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab50 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Input Image:} 
    vTcl:DefineAlias "$top.lab50" "LabelInputImg" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab53 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -justify right \
        -text {Enter Area Threshold:} 
    vTcl:DefineAlias "$top.lab53" "LabelAreaThresh" vTcl:WidgetProc "Toplevel1" 1
    button $top.but44 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text {Select Output Dir} 
    vTcl:DefineAlias "$top.but44" "ButtonIOutDir" vTcl:WidgetProc "Toplevel1" 1
    bind $top.but44 <ButtonRelease-1> {
        lambda e: btn_select_output_dir(e)
    }
    label $top.lab45 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Output Dir:} 
    vTcl:DefineAlias "$top.lab45" "LabelOutDir" vTcl:WidgetProc "Toplevel1" 1
    button $top.but47 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12 -weight bold} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text {SELECT STAINS} 
    vTcl:DefineAlias "$top.but47" "ButtonSelectStains" vTcl:WidgetProc "Toplevel1" 1
    bind $top.but47 <Button-1> {
        lambda e: btn_select_stains(e)
    }
    bind $top.but47 <ButtonRelease-1> {
        lambda e: btn_select_stains(e)
    }
    set site_3_0 $top.m48
    menu $site_3_0 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -font {-family {Arial} -size 12 -weight bold} \
        -foreground {#000000} -tearoff 0 
    $site_3_0 add command \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -command {#prog_descript} -font TkDefaultFont \
        -foreground {#000000} -label {Program Description} 
    $site_3_0 add separator \
        -background {#d9d9d9} 
    $site_3_0 add command \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -command {#prog_instruct} -font TkDefaultFont \
        -foreground {#000000} -label {Program Instructions} 
    $site_3_0 add separator \
        -background {#d9d9d9} 
    $site_3_0 add command \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -command {#quit} -font TkDefaultFont \
        -foreground {#000000} -label Quit 
    entry $top.ent43 \
        -background white -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -textvariable low_hue 
    vTcl:DefineAlias "$top.ent43" "EntryLowHue" vTcl:WidgetProc "Toplevel1" 1
    entry $top.ent45 \
        -background white -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -textvariable up_hue 
    vTcl:DefineAlias "$top.ent45" "EntryUpHue" vTcl:WidgetProc "Toplevel1" 1
    entry $top.ent46 \
        -background white -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -textvariable area_thresh 
    vTcl:DefineAlias "$top.ent46" "EntryAreaThresh" vTcl:WidgetProc "Toplevel1" 1
    entry $top.ent44 \
        -background white -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -textvariable low_sat 
    vTcl:DefineAlias "$top.ent44" "EntryLowSat" vTcl:WidgetProc "Toplevel1" 1
    entry $top.ent47 \
        -background white -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -textvariable low_val 
    vTcl:DefineAlias "$top.ent47" "EntryLowVal" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab49 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Hue 
    vTcl:DefineAlias "$top.lab49" "LabelHueLow" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab51 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Saturation 
    vTcl:DefineAlias "$top.lab51" "LabelSatLow" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab52 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Value 
    vTcl:DefineAlias "$top.lab52" "LabelValueLow" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab54 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Saturation 
    vTcl:DefineAlias "$top.lab54" "LabelUpSat" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab55 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Value 
    vTcl:DefineAlias "$top.lab55" "LabelUpVal" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab56 \
        -activebackground {#f9f9f9} -activeforeground black -anchor w \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Hue 
    vTcl:DefineAlias "$top.lab56" "LabelUpHue" vTcl:WidgetProc "Toplevel1" 1
    entry $top.ent57 \
        -background white -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -textvariable up_sat 
    vTcl:DefineAlias "$top.ent57" "EntryUpSat" vTcl:WidgetProc "Toplevel1" 1
    entry $top.ent58 \
        -background white -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -textvariable up_val 
    vTcl:DefineAlias "$top.ent58" "EntryUpVal" vTcl:WidgetProc "Toplevel1" 1
    entry $top.ent48 \
        -background white -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} \
        -insertbackground black -textvariable gaussian_blur 
    vTcl:DefineAlias "$top.ent48" "EntryGaussianBlur" vTcl:WidgetProc "Toplevel1" 1
    label $top.lab57 \
        -anchor w -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font {-family {Arial} -size 12} -foreground {#000000} -justify left \
        -text {Gaussian Blur (ODD Integers Only)} 
    vTcl:DefineAlias "$top.lab57" "LabelBlur" vTcl:WidgetProc "Toplevel1" 1
    ###################
    # SETTING GEOMETRY
    ###################
    place $top.but43 \
        -in $top -x 30 -y 560 -width 87 -relwidth 0 -height 44 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.but45 \
        -in $top -x 60 -y 20 -anchor nw -bordermode ignore 
    place $top.lab46 \
        -in $top -x 60 -y 110 -width 285 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab48 \
        -in $top -x 60 -y 200 -width 275 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab50 \
        -in $top -x 60 -y 60 -width 685 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab53 \
        -in $top -x 60 -y 340 -width 165 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.but44 \
        -in $top -x 60 -y 410 -width 141 -height 30 -anchor nw \
        -bordermode ignore 
    place $top.lab45 \
        -in $top -x 60 -y 450 -width 685 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.but47 \
        -in $top -x 220 -y 500 -width 158 -relwidth 0 -height 30 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.ent43 \
        -in $top -x 70 -y 160 -width 100 -relwidth 0 -height 22 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.ent45 \
        -in $top -x 70 -y 250 -width 100 -relwidth 0 -height 22 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.ent46 \
        -in $top -x 70 -y 370 -width 100 -relwidth 0 -height 22 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.ent44 \
        -in $top -x 190 -y 160 -width 100 -relwidth 0 -height 22 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.ent47 \
        -in $top -x 310 -y 160 -width 100 -relwidth 0 -height 22 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab49 \
        -in $top -x 70 -y 130 -anchor nw -bordermode ignore 
    place $top.lab51 \
        -in $top -x 190 -y 130 -anchor nw -bordermode ignore 
    place $top.lab52 \
        -in $top -x 310 -y 130 -anchor nw -bordermode ignore 
    place $top.lab54 \
        -in $top -x 190 -y 220 -width 84 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab55 \
        -in $top -x 310 -y 220 -width 54 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab56 \
        -in $top -x 70 -y 220 -width 34 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.ent57 \
        -in $top -x 190 -y 250 -width 100 -height 22 -anchor nw \
        -bordermode ignore 
    place $top.ent58 \
        -in $top -x 310 -y 250 -width 100 -height 22 -anchor nw \
        -bordermode ignore 
    place $top.ent48 \
        -in $top -x 70 -y 310 -width 100 -relwidth 0 -height 22 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab57 \
        -in $top -x 60 -y 280 -width 245 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 

    vTcl:FireEvent $base <<Ready>>
}

set btop ""
if {$vTcl(borrow)} {
    set btop .bor[expr int([expr rand() * 100])]
    while {[lsearch $btop $vTcl(tops)] != -1} {
        set btop .bor[expr int([expr rand() * 100])]
    }
}
set vTcl(btop) $btop
Window show .
Window show .top42 $btop
if {$vTcl(borrow)} {
    $btop configure -background plum
}
