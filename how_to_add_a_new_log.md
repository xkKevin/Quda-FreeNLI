# How to add a new log to FreeNLI



### API - create a new log file

URL: `https://freenli.projects.zjvis.org/create_log/`

Methods: `GET`

Params:

+ type: your new log_file name:  log_`type`.csv
+ property: an array that showcases the fields of the log_file
  + *default*: `'["Time", "Ip", "Id", "Location"]'`
  + *note that the first field must be* `"Time"`
+ property_type: the type of each field
  + 0 → categorical
  + 1 → quantitative
  + *default*: `'[0, 0, 1, 0]'`
+ force: if the log_file is already existed, you can force to recreate it.
  + 0 (*default*) → false
  + 1 → true

Example: `https://freenli.projects.zjvis.org/create_log/?type=test`



### API - recode logs to the log file

URL: `https://freenli.projects.zjvis.org/access_log/`

Methods: `POST`

Params:

+ ip: the ip address of the web user
+ id: the number of Zip code
+ location: location info, such as province, city, and country
+ type: your log_file name:  log_`type`.csv

 *NOTE: the order of those parameters must be consistent with the order of  the fields in log_file, except the 'type'*

Example:

Note: you should introduce this script to your `<head>`：`<script src="https://pv.sohu.com/cityjson?ie=utf-8"></script>`

```javascript
// jquery
$(function () {
    const path = "https://freenli.projects.zjvis.org/access_log/"
    const data = {
        "ip":returnCitySN.cip,
        "id":returnCitySN.cid, 
        "location":returnCitySN.cname, 
        'type': 'test'
    }
    $.post(path, data, function(result, statue){
        if (!result.result){
            console.log(result.error);
        }
    });
}

// axios 
mount(){
    const path = 'https://freenli.projects.zjvis.org/access_log/'
    let data = new FormData();
    data.append('ip', returnCitySN.cip);
    data.append('id',returnCitySN.cid);
    data.append('location',returnCitySN.cname);
    data.append('type','test');
    axios
      .post(path, data)
      .then((response) => {
        if (!response.data.result){
            console.log(response.data.error);
        }
      })
      .catch((error) => {
        console.log(error);
      });
}
```

[axis传参](https://segmentfault.com/a/1190000015261229?utm_source=tag-newest)



### API - delete a log file

URL: `https://freenli.projects.zjvis.org/delete_log/`

Methods: `GET`

Params:

+ type: your log_file name:  log_`type`.csv

Example: `https://freenli.projects.zjvis.org/delete_log/?type=test`


##### Others
Modify access log file:
`kubectl cp log_test.csv freenli-c794856f6-tkqds:/freeNLI/main/static/table/log_test.csv`