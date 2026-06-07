# Emb_VQVAE




| 特征名称       | 溯源pb字段                                                                 | 当前ClassName                       |
| -------------- | ------------------------------------------------------------------------ | ----------------------------------- |
| 用户ID         | user_info.id                                                            | ExtractUserId                      |
| 所在省市       | - user_info.long_term_loc.city                                           | ExtractUserLoc                     |
|                | - user_info.long_term_loc.province                                      |                                     |
| 消费等级       | user_info.ad_user_info.consumption_level                                | ExtractUserConsumptionLevel        |
| 商业行业兴趣   | user_info.ad_user_info.business_interest                                | ExtractUserBusinessInterest        |
| 手机信息       | - user_info.ad_user_info.device_info                                    | ExtractUserAdDeviceInfo            |
|                | - user_info.ad_user_info.platform                                       | ExtractUserDeviceInfoAdUser        |
|                | - user_info.ad_user_info.platform_version                               | ExtractUserDeviceInfoNew           |
|                | - user_info.ad_user_info.network                                        | ExtractUserDeviceInfoNewhash       |
|                | - user_info.device_info.visit_net                                       |                                     |
|                | - user_info.device_info.visit_mod                                       |                                     |
| 设备ID（可视为用户ID的另一表现形式） | - user_info.ad_user_info.device_info.idfa                         | ExtractUserAdDeviceInfoImei         |
|                | - user_info.ad_user_info.device_info.imei                               | ExtractUserDeviceIdNew             |
|                | - user_info.ad_user_info.device_info.device_id                          |                                     |
| 安装APP列表    | user_info.ad_user_info.device_info.app_package                          | ExtractUserAppList                 |
| 人生阶段       | - user_info.ad_user_info.has_car                                        | ExtractUserLifeStat                |
|                | - user_info.ad_user_info.has_house                                      |                                     |
|                | - user_info.ad_user_info.is_work                                        |                                     |
|                | - user_info.ad_user_info.life_stage                                     |                                     |
| 年龄&等级      | - user_info.ad_user_info.age                                            | ExtractUserAttributeNew            |
|                | - user_info.level                                                      |                                     |


| 特征名称       | 溯源pb字段                           | 当前ClassName            |
| -------------- | ------------------------------------ | ------------------------ |
| follow用户量   | user_info.attribute.follow_count     | ExtractUserCountFeature  |
| 上传作品量     | user_info.attribute.upload_count     | ExtractUserCountFeature  |
