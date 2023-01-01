function detectColumns(data, columns_name, columns_type) {
    /**
     * @return: C, N, NC, CC, NN  --> 1, 2, 3, 4, 5
     */
    if (columns_type.length===1){
        return columns_type[0] ? 2 : 1;
    }
    if (columns_type.length===2) {
        if (columns_type[0]){
            return columns_type[1] ? 5 : 3;
        }
        if (columns_type[1]){
            // swapColumns(data, columns_name, columns_type, 1);
            moveColumns(data, columns_name, columns_type, 1);
            return 3;
        }
        return 4;
    }
    if (columns_type.length >= 3){  // 大于等于三列的情况
        let fnp = columns_type.indexOf(1); // fnp: first_numerical_position，首先出现数字类型的位置
        switch (fnp) {
            case -1: return 4;
            case 0: return columns_type[1] ? 5 : 3;
            default: moveColumns(data, columns_name, columns_type, fnp); return 3;
        }
    }
}


function judgeBar(data, columns_type) {
    // 当数据只有一列，或有多列但都是C类型，或有N类型但不成函数关系（不满足：给定一个x，有且只有一个y值），则不能绘制bar chart
    if (columns_type.length===1){
        return false;
    }
    if (columns_type[0] + columns_type[1] === 0){
        return false;
    }
    let data_first_set = [];
    for (let i = 0; i < data.length; i++) {
        if (data_first_set.includes(data[i][0])){
            return false;
        }else{
            data_first_set.push(data[i][0]);
        }
    }
    return true;
}


function moveColumns(data, columns_name, columns_type, position) {
    // 将下标为 position 的元素与 移至数组最前面
    columns_name.unshift(columns_name.splice(position,1)[0]);
    columns_type.unshift(columns_type.splice(position,1)[0]);
    data.map((v)=> v.unshift(v.splice(position,1)[0]));
}


function unshiftColumn(data, addColumn=0, columns_name=0) {
    // data 和 addColumn 数组长度必须一样
    if (addColumn === 0){
        addColumn = Array.apply(null, Array(data.length)).map(() => 1);
    }
    for (let j = 0; j < data.length; j++){
        data[j].unshift(addColumn[j]);
    }
    if (columns_name !== 0){
        columns_name.unshift("Num");
    }
}


function binStragegy(data, columns_name, columns_type, bin=10) {
    let dmin, dmax;
    dmin = dmax = data[0][0];
    for (let i = 1; i < data.length; i++) {
        if (data[i][0] < dmin){
            dmin = data[i][0];
        }else if (data[i][0] > dmax){
            dmax = data[i][0];
        }
    }
    let bin_interval = (dmax - dmin) / bin;
    let data_n = [], data_b = [], result = [];
    for (let i = 0; i < bin; i++) {
        result.push([]);
        data_n.push(0);
        data_b.push(parseFloat((dmin+i*bin_interval).toFixed(2)) + "-" + parseFloat((dmin+(i+1)*bin_interval).toFixed(2)));
    }
    for (let i = 0; i < data.length; i++) {
        let bp = Math.floor((data[i][0] - dmin)/bin_interval);
        if (bp < bin){
            data_n[bp]++;
            result[bp].push(data[i].slice(1));
        }else{
            data_n[bin-1]++;
            result[bin-1].push(data[i].slice(1));
        }
    }
    unshiftColumn(result,data_b);
    unshiftColumn(result,data_n);
    columns_name[0] = columns_name[0] + " Interval";
    columns_name.unshift("Num");
    columns_type[0] = 0;
    columns_type.unshift(1);
    return result;
}


function countStrategy(data, columns_name, columns_type) {
    let data_c = [], data_n = [], result = [];
    data.map(function (v){
        let pi = data_c.indexOf(v[0]);
        if (pi === -1){
            data_c.push(v[0]);
            data_n.push(1);
            result.push([v.slice(1)]);
        }else{
            data_n[pi]++;
            result[pi].push(v.slice(1));
        }
    });
    unshiftColumn(result,data_c);
    unshiftColumn(result,data_n);
    columns_name.unshift("Count");
    columns_type.unshift(1);
    return result;
}


function sortStrategy(data) {
    data.sort(cp);
}


function cp(f,c){
    let g,h;
    if (f.length === undefined){
        g=f; h=c;
    }else{
        g=f[0]; h=c[0];
    }
    let reg = /^[-+]?([1-9][0-9]*|0)(\.([0-9]+))?$/;
    // console.log(g,h);
    let i,n;
    if (reg.test(g) && reg.test(h)){
        i=parseFloat(g); n=parseFloat(h);
        g=i; h=n
        // if(!isNaN(i)&&!isNaN(n)){g=i; h=n}
    }
    else {
        i=Date.parse(g); n=Date.parse(h);
        if(!isNaN(i)&&!isNaN(n)){g=i; h=n}
    }
    return g>h?1:(g<h?-1:0)
}


function swapColumns(data, columns_name, columns_type, position) {
    // 将下标为 position 的元素与 0 元素交换
    let tmp;
    tmp = columns_name[0];
    columns_name[0] = columns_name[position];
    columns_name[position] = tmp;

    tmp = columns_type[0];
    columns_type[0] = columns_type[position];
    columns_type[position] = tmp;

    for (let j = 0; j < data.length; j++){
        tmp = data[j][0];
        data[j][0] = data[j][position];
        data[j][position] = tmp;
    }
}


function sum(arr) {
    var s = 0;
    for (var i=arr.length-1; i>=0; i--) {
        s += arr[i];
    }
    return s;
}