var chartType = ["scatter", "bar", "line", "pie"];
var tooltipShowNum = 10;  // 10 - 2 = 8
function draw_charts(data, columns_name, columns_type, charts, task, task_index=0) {
    let charts_div = $("#charts");
    if (task_index === 0){
        charts_div.empty();
    }
    let data_type = detectColumns(data, columns_name, columns_type);
    // console.log(task,data_type);
    let data_a = data,
        cn_a = JSON.parse(JSON.stringify(columns_name)),  // columns_name in all charts (Scatter, bar, line chart)
        ct_a = columns_type;   // columns_type in all charts
    let data_p = JSON.parse(JSON.stringify(data)),
        cn_p = JSON.parse(JSON.stringify(columns_name)),
        ct_p = JSON.parse(JSON.stringify(columns_type));

    let mark_line=null, mark_point=null;

    switch (data_type) {
        case 1:
            switch (task) {
                case 0:
                case 1:
                case 2:
                case 5:
                case 10:
                    sortStrategy(data_a);
                    sortStrategy(data_p);
                    unshiftColumn(data_p, 0, cn_p);
                    break;
                case 3:
                case 4:
                case 6:
                case 7:
                case 8:
                case 9:
                    data_a = countStrategy(data_a, cn_a, ct_a);
                    swapColumns(data_a, cn_a, ct_a, 1);
                    sortStrategy(data_a);
                    data_p = countStrategy(data_p, cn_p, ct_p);
                    sortStrategy(data_p);
                    break;
            }
            break;
        case 2:
            switch (task) {
                case 0:
                case 1:
                case 2:
                case 4:
                case 5:
                case 6:
                case 10:
                    sortStrategy(data_a);
                    sortStrategy(data_p);
                    break;
                case 3:
                case 7:
                case 8:
                case 9:
                    data_a = binStragegy(data_a, cn_a, ct_a);
                    swapColumns(data_a, cn_a, ct_a, 1);
                    data_p = binStragegy(data_p, cn_p, ct_p);
                    break;
            }
            break;
        case 3:
            switch (task) {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 6:
                case 8:
                case 9:
                case 10:
                    swapColumns(data_a, cn_a, ct_a, 1);
                    sortStrategy(data_a);
                    sortStrategy(data_p);
                    break;
                case 5:
                    sortStrategy(data_a);
                    swapColumns(data_a, cn_a, ct_a, 1);
                    sortStrategy(data_p);
                    break;
                case 7:
                    data_a = binStragegy(data_a, cn_a, ct_a);
                    swapColumns(data_a, cn_a, ct_a, 1);
                    data_p = binStragegy(data_p, cn_p, ct_p);
                    break;
            }
            break;
        case 4:
            switch (task) {
                case 0:
                case 1:
                case 2:
                case 5:
                case 8:
                case 9:
                case 10:
                    sortStrategy(data_a);
                    sortStrategy(data_p);
                    unshiftColumn(data_p, 0, cn_p);
                    break;
                case 3:
                case 4:
                case 6:
                case 7:
                    data_a = countStrategy(data_a, cn_a, ct_a);
                    swapColumns(data_a, cn_a, ct_a, 1);
                    sortStrategy(data_a);
                    data_p = countStrategy(data_p, cn_p, ct_p);
                    sortStrategy(data_p);
                    break;
            }
            break;
        case 5:
            switch (task) {
                case 0:
                case 1:
                case 2:
                case 4:
                case 5:
                case 6:
                case 8:
                case 9:
                case 10:
                    sortStrategy(data_a);
                    sortStrategy(data_p);
                    break;
                case 3:
                case 7:
                    data_a = binStragegy(data_a, cn_a, ct_a);
                    swapColumns(data_a, cn_a, ct_a, 1);
                    data_p = binStragegy(data_p, cn_p, ct_p);
                    break;
            }
            break;
        default :
            alert("Error data type!" + data_type);
    }

    let draw_max_min_avg_tasks = [0,3,4];
    if (draw_max_min_avg_tasks.includes(task)){  // Others, Compute Derived Value, Find Extremum
        if (sum(ct_a) > 0){  // 类型为numerical
            mark_line = [{type : 'average', name: 'average'}];
            mark_point = [{type: 'min', name: 'min'},{type: 'max', name: 'max'}];
        }
    }else if (task === 5){
        if (sum(ct_a) > 0){  // 类型为numerical
            mark_line = [{type : 'average', name: 'average'}];
        }
    }

    let draw_mean_tasks = [6, 10];
    let mean_points = [];
    if (draw_mean_tasks.includes(task)){  // Determine Range, Correlate
        if (sum(ct_a) > 0){  // 类型为numerical
            let data_dict = {};
            data_a.forEach(function (x){
                if (data_dict[x[0]] === undefined){
                    data_dict[x[0]] = [x[1]];
                }else{
                    data_dict[x[0]].push(x[1]);
                }
            });
            for (let i in data_dict){
                if (data_dict[i].length > 1){
                    let mean = sum(data_dict[i])/data_dict[i].length;
                    mean_points.push({
                        xAxis: i,
                        yAxis: mean,
                        name: 'mean',
                        value: mean,
                    })
                }
            }
            if (mean_points.length){
                mark_point = {
                    symbol: 'rect',
                    // symbolRotate: -90,
                    symbolSize: [12,2], // 长、高
                    label: {
                      formatter: '', //{c}
                      color: "black",
                      offset: [6,5],
                      fontSize: 12
                    },
                    itemStyle: {
                      color: "white",
                      borderColor: '#1e90ff'
                    },
                    data: mean_points
                };
            }
        }
    }

    for (let ci=0; ci<charts.length; ci++){
        if (charts[ci] === 1 && !judgeBar(data_a, ct_a) && !mean_points.length){
            console.log("Nothing can show for " + task + " type task.");
            continue;
        }
        let chart_title_id = 'chart_title' + task_index + ci;
        let chart_body_id = 'chart_body' + task_index + ci;
        let div_chart =
            `<div class="vis_chart">
                <div class="panel" style="border-color: rgb(226,227,255)">
                    <div class="panel-heading" id=${chart_title_id}>
                    </div>
                    <div class="panel-body" id=${chart_body_id} style="height: 380px;">
                    </div>
                </div>
            </div>`;

        charts_div.append(div_chart);
        if (charts[ci] === 1 && !judgeBar(data_a, ct_a)){   // 当为bar chart且一个x对应多个y的情况
            // console.log(data_a, cn_a, ct_a, mark_point);
            bar_mean_chart(chart_body_id, cn_a, mark_point.data);
        }else if (charts[ci] === 3){
            pie_chart(chart_body_id, data_p, cn_p, ct_p);  // data_p.slice(0,20) 只显示前20条记录
        }else{
            line_bar_scatter_chart(chart_body_id, data_a, cn_a, ct_a, chartType[charts[ci]], mark_line, mark_point);
        }
    }
    $(".vis_chart .panel-body").css("height", "356px");
}


function bar_mean_chart(chart_id, columns_name, data) {
    // $("#chart_title"+chart_id).text("bar chart");
    let dom = document.getElementById(chart_id);  // "chart_body" + chart_id
    echarts.dispose(dom);  // 不能在单个容器上初始化多个 ECharts 实例。
    let myChart = echarts.init(dom);
    let xdata = [], ydata = [], max = 0;
    data.forEach(function (d) {
       xdata.push(d.value);
       ydata.push(d.xAxis);
       if (max < d.xAxis.length) {
           max = d.xAxis.length
       }
    });
    let option = {
        tooltip:{
            formatter: function (param) {
                return `Mean ${columns_name[1]}<br>${param.name}: `+parseFloat(param.value.toFixed(2));
            }
        },
        yAxis: {
            type: 'category',
            data: ydata,
            name: columns_name[0],
            nameTextStyle: {
                fontWeight: 'bold'
            }
        },
        xAxis: {
            type: 'value',
            name: columns_name[1],
            nameGap: columns_name[1].length>8?40:5,
            nameRotate: columns_name[1].length>8?-90:0,
            nameTextStyle: {
                align: columns_name[1].length>8?'right':'left',
                fontWeight: 'bold'
            },
            axisLabel:{
                formatter: formatter_label
            }
        },
        grid:{
          left: 28 + max * 6
        },
        series: [{
            data: xdata,
            type: 'bar',
            label: {
                normal: {
                    show: true,
                    position: 'right',
                    color: 'black',
                    formatter: function (param) {
                        let value = param.value;
                        if (value >= 1000000000){
                            return parseFloat((value/1000000000).toFixed(2)) + "B"
                        }else if (value >= 1000000){
                            return parseFloat((value/1000000).toFixed(2)) + "M"
                        }if (value >= 1000){
                            return parseFloat((value/1000).toFixed(2)) + "K"
                        }
                        return parseFloat(value.toFixed(2))
                    }
                }
            },
        }]
    };
    if (option && typeof option === "object") {
        myChart.setOption(option, true);
    }
}


function line_bar_scatter_chart(chart_id, data, columns_name, columns_type, chart_type, mark_line=null, mark_point=null) {
    /*
        mark_line: 对象数组，[{type : 'average', name: 'average'}]
        mark_point: 对象数组，[{type: 'min', name: 'min'},{type: 'max', name: 'max'}]
     */
    // $("#chart_title"+chart_id).text(chart_type + " chart");
    let dom = document.getElementById(chart_id);  // "chart_body"+
    echarts.dispose(dom);  // 不能在单个容器上初始化多个 ECharts 实例。
    let myChart = echarts.init(dom);

    let option = {
        // title: {text: chart_type + " chart"},
        tooltip: {
            // position: 'right',
            formatter: function(params){
                if (params.data[0] === undefined){ // markLine, markPoint
                    return params.data.name + ": " + params.data.value;
                }
                let ttext = columns_name[0]+ ": " + params.data[0];
                if (params.data[1] || params.data[1] === 0){
                    ttext += "<br>"+columns_name[1]+ ": " + params.data[1];
                    if (columns_name.length > 2){
                        ttext += "<br>" + columns_name.slice(2).join(", ");
                    }
                    let data_others_len = data[params.dataIndex].length > tooltipShowNum ? tooltipShowNum : data[params.dataIndex].length;
                    for (let i = 2; i < data_others_len; i++) {
                        if (data[params.dataIndex][i] instanceof Array){  // 字符串不是数组
                            if (data[params.dataIndex][i].length){
                                ttext += "<br>" + data[params.dataIndex][i].join(", ");
                            }
                        }else {
                            if (i === 2) ttext += "<br>" + data[params.dataIndex][i];
                            else ttext += ", " + data[params.dataIndex][i];
                        }
                    }
                }
                return ttext;
            }
        },
        grid: {
        },
        xAxis: {
            type: columns_type[0]?"value":'category',
            min: function (value) {
                let min_tmp = value.min - Math.ceil(Math.abs(value.min)*0.05);
                if ((value.min - min_tmp) > (value.max - value.min)/2)
                    return value.min;
                return value.min - Math.ceil(Math.abs(value.min)*0.05);
            },
            max: 'dataMax',
            axisLine: {onZero: false},
            name: columns_name[0],
            nameGap: columns_name[0].length>8?30:5,
            nameRotate: columns_name[0].length>8?-90:0,
            nameTextStyle: {
                align: columns_name[0].length>8?'right':'left',
                fontWeight: 'bold'
            }
        },
        yAxis: {
            type: columns_type[1]?"value":'category',
            min: function (value) {
                let min_tmp = value.min - Math.ceil(Math.abs(value.min)*0.05);
                if ((value.min - min_tmp) > (value.max - value.min)/2)
                    return value.min;
                return value.min - Math.ceil(Math.abs(value.min)*0.05);
            },
            max: 'dataMax',
            axisLine: {onZero: false},
            name: columns_name[1],
            // nameGap: 15,  // 默认是 14
            nameTextStyle: {
                fontWeight: 'bold'
            },
            axisLabel:{
                formatter: formatter_label
            }
        },
        series: [
            {
                type: chart_type,
                label: {
                    normal: {
                        show: chart_type === "bar",
                        position: 'top',
                        color: 'black',
                        formatter: formatter_label
                    }
                },
                smooth: true,
                data: data,  // 0号元素做x轴
                markLine : {
                    itemStyle : {
                        normal : {
                            color:'#1e90ff'
                        }
                    },
                    data: mark_line
                },
                markPoint: mark_point === null || chart_type === 'bar' ? null: mark_point['symbol'] === undefined ? {
                    itemStyle : {
                        normal : {
                            color:'#1e90ff'
                        }
                    },
                    data: mark_point
                } : null // mark_point,
            }
        ]
    };
    if (option && typeof option === "object") {
        myChart.setOption(option, true);
    }
}


function pie_chart(chart_id, data, columns_name, columns_type) {
    // $("#chart_title"+chart_id).text("pie chart");
    let dom = document.getElementById(chart_id);  // "chart_body"+
    echarts.dispose(dom);  // 不能在单个容器上初始化多个 ECharts 实例。
    let myChart = echarts.init(dom);
    let show_label = data.length <= 12;
    let option = {
        tooltip: {
            trigger: 'item',
            formatter: show_label ? '{b} : {c} ({d}%)': function (params){
                let ttext = params.seriesName + ", Percent<br>";
                ttext += params.data.slice(0,2).join(", ") + ", " + params.percent + "%";
                if (columns_name.length > 2){
                    ttext += "<br>" + columns_name.slice(2).join(", ");
                }
                let data_others_len = data[params.dataIndex].length > tooltipShowNum ? tooltipShowNum : data[params.dataIndex].length;
                for (let i = 2; i < data_others_len; i++) {
                    if (data[params.dataIndex][i] instanceof Array){  // 字符串不是数组
                        if (data[params.dataIndex][i].length){
                            ttext += "<br>" + data[params.dataIndex][i].join(", ");
                        }
                    }else {
                        if (i === 2) ttext += "<br>" + data[params.dataIndex][i];
                        else ttext += ", " + data[params.dataIndex][i];
                    }
                }
                return ttext;
            }
        },
        legend:{
            show: show_label,
            orient: 'vertical',
            right: 10,
            top: 'center',
            data: data.map((x)=> x[1]).reverse()
        },
        series: [
            {
                name: columns_name.slice(0,2).join(", "),
                type: 'pie',
                radius: show_label ? '55%': '80%',
                center: ['35%', '50%'],
                data: show_label ? data.map((x)=>{ return {value:x[0],name:x[1]} }) : data,
                label: {
                    show: show_label
                }
            }
        ]
    };

    if (option && typeof option === "object") {
        myChart.setOption(option, true);
    }
}

function formatter_label(param) {
    // 数字的话使用k，m，b简化0，长字符串使用首字母
    if (!isNaN(param)){
        if (param >= 1000000000){
            return parseFloat((param/1000000000).toFixed(2)) + "B"
        }else if (param >= 1000000){
            return parseFloat((param/1000000).toFixed(2)) + "M"
        }if (param >= 1000){
            return parseFloat((param/1000).toFixed(2)) + "K"
        }
        return parseFloat(param.toFixed(2))
    }else if (typeof(param) === "object"){
        param = param.value[1];
        if (param >= 1000000000){
            return parseFloat((param/1000000000).toFixed(2)) + "B"
        }else if (param >= 1000000){
            return parseFloat((param/1000000).toFixed(2)) + "M"
        }if (param >= 1000){
            return parseFloat((param/1000).toFixed(2)) + "K"
        }
        return parseFloat(param.toFixed(2))
    }else if (typeof(param) === "string"){
        if (param.length > 7){
            return param.split(" ").map((x)=>x[0]).join(""); // 返回每个单词的首字母
        }
        return param;
    }
    return ""
}